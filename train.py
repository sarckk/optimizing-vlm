import os
import contextlib
import time
from omegaconf import DictConfig
import wandb
from datetime import datetime
import hydra
from typing import List, Optional, Union
import torch

from torch.utils.data import (
    DataLoader,
    IterableDataset,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data import DataConfig, get_dataloaders
from modules.vision_language_model import VisionLanguageModel

from configs import TrainerConfig, ModelConfig, DataConfig, ProfilerMode

SEED = 1337
CUDA_GRAPH_WARMUP_ITERS = 3
CUDA_GRAPH_WARMUP_ITERS_DDP = 11
SNAPSHOT_FILE_FORMAT = "{0:02d}_snapshot.pt"
PROFILE_TRACE_FORMAT = "trace_step_{0}_{1}.json.gz"
PROFILE_MEMORY_SNAPSHOT_FORMAT = "mem_snapshot_{0}.html"

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def run_validation(
    model, mp_context, epoch, global_step, trainer_config, val_dataloader, device
):
    print(f"Epoch {epoch} running validation at global step {global_step}")
    losses = []

    model.eval()

    for step, (images, captions, targets) in enumerate(val_dataloader):
        if step == trainer_config.eval_iters:
            break

        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)

        # Disable gradient calculation
        with torch.no_grad():
            with mp_context:
                _, loss = model(images, captions, targets)
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)

    print(f"Validation Loss after epoch {epoch}: {val_loss}")

    if trainer_config.use_wandb:
        wandb.log({"validation/loss": val_loss, "global_step": global_step})


def save_snapshot(model, optimizer, scaler, trainer_config, epoch, wandb_run_id):
    snapshot_dir = os.path.join(os.path.dirname(__file__), trainer_config.snapshot_dir)
    if not os.path.exists(snapshot_dir):
        # Create directory if not exists
        os.makedirs(snapshot_dir)

    assert os.path.isdir(
        snapshot_dir
    ), f"Expected {snapshot_dir} to be a directory but is not."

    snapshot_path = os.path.join(snapshot_dir, SNAPSHOT_FILE_FORMAT.format(epoch))

    snapshot = {
        "model_state": (
            model.module.state_dict() if trainer_config.use_ddp else model.state_dict()
        ),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "wandb_run_id": wandb_run_id,
        "epochs_run": epoch + 1,  # e.g. after epoch 0, we have run 1 epoch
    }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Saved training snapshot at {snapshot_path}")


def export_trace(profile_results_dir, profiler_mode, prof):
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    step_num = str(prof.step_num)
    trace_path = os.path.join(
        profile_results_dir, PROFILE_TRACE_FORMAT.format(step_num, timestamp)
    )
    prof.export_chrome_trace(trace_path)

    try:
        print(f"Exported trace to {trace_path}")
    except Exception as e:
        print(
            f"Failed to save trace to {memory_snapshot_path}. If num steps run < 10, try increasing the step count. Error: {e}"
        )
        return

    if profiler_mode and profiler_mode == ProfilerMode.MEMORY.value:
        memory_snapshot_path = os.path.join(
            profile_results_dir,
            PROFILE_MEMORY_SNAPSHOT_FORMAT.format(timestamp),
        )
        try:
            prof.export_memory_timeline(memory_snapshot_path)
            print(f"Saved memory history snapshot to {memory_snapshot_path}")
        except Exception as e:
            print(
                f"Failed to save memory history snapshot to {memory_snapshot_path}: {e}"
            )
            return


def init_wandb(
    wandb_run_id: Optional[str],
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
):
    wandb.init(
        project=trainer_config.wandb_project,
        resume="allow",  # Allow wandb to resume run, by defalut it overrides
        id=wandb_run_id,
        config={**model_config.__dict__, **trainer_config.__dict__},
    )
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")


def get_profiler_context(profiler_mode: Optional[ProfilerMode], master_process: bool):
    should_profile = master_process and profiler_mode is not None
    profile_memory = profiler_mode == ProfilerMode.MEMORY.value
    # Creates no-op context if 'trace' not in requested profiler modes
    profiler_context = (
        contextlib.nullcontext()
        if not should_profile
        else torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=profile_memory,
            record_shapes=profile_memory,
            with_stack=profile_memory,
            schedule=torch.profiler.schedule(
                skip_first=1,
                wait=1,
                warmup=3,
                active=3,
                repeat=1,
            ),
        )
    )
    return profiler_context, should_profile


def find_and_load_snapshot(snapshot_dir: str, snapshot_load_strategy: Union[str, int]):
    assert os.path.isdir(
        snapshot_dir
    ), f"Expected {snapshot_dir} passed in to be a directory"

    if snapshot_load_strategy == "latest":
        # All files should be named as {epoch}_snapshot.pt, fetch the file with the latest epoch
        sorted_snapshot_files: List[str] = sorted(
            [
                f
                for f in os.listdir(snapshot_dir)
                if os.path.isfile(os.path.join(snapshot_dir, f)) and f.endswith(".pt")
            ],
            key=lambda f: int(f.split("_")[0]),
        )
        if len(sorted_snapshot_files) == 0:
            print(
                f"Could not find any snapshot files in dir {snapshot_dir}. Will not load any snapshots"
            )
            return None

        latest_snapshot_file = sorted_snapshot_files[-1]
        return torch.load(os.path.join(snapshot_dir, latest_snapshot_file))

    elif isinstance(snapshot_load_strategy, int):
        # Load from specific epoch
        epoch_to_load = snapshot_load_strategy
        print(f"Loading snapshot from epoch {epoch_to_load}...")
        snapshot_path = os.path.join(
            snapshot_dir, SNAPSHOT_FILE_FORMAT.format(epoch_to_load)
        )
        assert os.path.exists(
            snapshot_path
        ), f"Could not find snapshot file {snapshot_path} to load!"
        return torch.load(snapshot_path)
    else:
        raise Exception(
            f"Snapshot load strategy of {snapshot_load_strategy} is not valid!",
            "Load strategy can either be 'null', 'latest' or a number representing the epoch to load from.",
        )


def compile_cuda_graph(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mp_context: Union[torch.autocast, contextlib.AbstractContextManager],
    train_dataloader: DataLoader,
    device: torch.device,
    use_ddp: bool,
    local_rank: int,
):
    print("CUDA graph enabled. Capturing graph...")
    dataloader_iter = iter(train_dataloader)
    # (images, captions, targets) tuple
    static_inputs = tuple(v.clone().to(device) for v in next(dataloader_iter))

    # Warmup for cuda graph capture
    s = torch.cuda.Stream()

    # Requires 11 DDP-enabled eager iterations before graph capture
    # from: https://pytorch.org/docs/main/notes/cuda.html#usage-with-distributeddataparallel
    num_iters_warmup = (
        CUDA_GRAPH_WARMUP_ITERS_DDP if use_ddp else CUDA_GRAPH_WARMUP_ITERS
    )

    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if use_ddp:
            # When using DDP + CUDA Graph, DDP must be created in the side stream
            assert not isinstance(
                model, DDP
            ), f"Expected model to not be DDP wrapped previously"
            model = DDP(model, device_ids=[local_rank])

        for i in range(num_iters_warmup):
            with mp_context:
                _, loss = model(*static_inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()

    # important: ensure backward() will create .grad attrs with allocations from graph's private pool
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_logits, static_loss = model(*static_inputs)
        static_loss.backward()
        optimizer.step()

    def replay_graph(
        images: torch.Tensor, captions: torch.Tensor, targets: torch.Tensor
    ):
        static_inputs[0].copy_(images)
        static_inputs[1].copy_(captions)
        static_inputs[2].copy_(targets)
        graph.replay()
        return static_logits, static_loss

    return replay_graph


def _wait_batch(batch, stream):
    if stream is None:
        # nothing to do if not on GPU
        return

    cur_stream = torch.cuda.current_stream()
    cur_stream.wait_stream(stream)
    for v in batch:
        v.record_stream(cur_stream)


def maybe_load_snapshot(
    trainer_config: TrainerConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    master_process: bool,  # am I in master process?
):
    start_epoch = 0
    wandb_run_id = None

    if os.path.isdir(trainer_config.snapshot_dir):
        if trainer_config.snapshot_load_strategy is not None:
            snapshot = find_and_load_snapshot(
                trainer_config.snapshot_dir,
                trainer_config.snapshot_load_strategy,
            )
            model.load_state_dict(snapshot["model_state"])
            optimizer.load_state_dict(snapshot["optimizer_state"])
            scaler.load_state_dict(snapshot["scaler_state"])
            # Resume wandb logging from previous snapshot
            wandb_run_id = snapshot["wandb_run_id"]
            start_epoch = int(snapshot["epochs_run"])
            if master_process:
                print(f"Resuming training from epoch {start_epoch}")
        elif master_process:
            print(
                f"Snapshot load strategy is 'null'. Skipping loading previous snapshot."
            )
    else:
        if master_process:
            print(
                f"Couldn't find snapshot dir: {trainer_config.snapshot_dir}. Will not resume from checkpoint"
            )

    return start_epoch, wandb_run_id


def get_mixed_precision_context_and_scaler(device: str, use_amp: bool):
    enable_grad_scaling = False

    if device == "cpu":
        # mixed precision not relevant for CPU
        ctx = contextlib.nullcontext()
    else:
        assert torch.cuda.is_available()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if use_amp:
            print(f"Using mixed precision with {dtype}")
        ctx = torch.amp.autocast(device_type=device, dtype=dtype, enabled=use_amp)
        # No need for gradient scaling by default for bfloat16
        enable_grad_scaling = dtype == torch.float16

    # if enabled=False, grad scaler is no-op
    scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaling and use_amp)
    return ctx, scaler


def train_model(
    model_config: ModelConfig, trainer_config: TrainerConfig, data_config: DataConfig
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_process = local_rank == 0

    if master_process:
        os.makedirs(trainer_config.profile_results_dir, exist_ok=True)

    tokenizer, train_dataloader, val_dataloader, train_sampler = get_dataloaders(
        model_config.img_size,
        trainer_config,
        data_config,
    )

    model = get_model(
        model_config, trainer_config, tokenizer.get_vocab_size(), device, local_rank
    )
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.lr,
        # Make optimizer step capturable by cuda graph
        capturable=trainer_config.cuda_graph,
    )

    profile_context, should_profile = get_profiler_context(
        trainer_config.profiler_mode, master_process
    )
    mp_context, scaler = get_mixed_precision_context_and_scaler(
        device, trainer_config.use_amp
    )

    start_epoch, wandb_run_id = maybe_load_snapshot(
        trainer_config, model, optimizer, scaler, master_process
    )

    if trainer_config.cuda_graph:
        compiled_model = compile_cuda_graph(
            model,
            optimizer,
            mp_context,
            train_dataloader,
            device,
            trainer_config.use_ddp,
            local_rank,
        )

    if trainer_config.use_wandb and master_process:
        init_wandb(wandb_run_id, model_config, trainer_config)
        if master_process:
            print("wandb logging enabled")

    memcpy_stream = (
        torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
    )

    global_step = 0

    print("starting training")

    for epoch in range(start_epoch, trainer_config.epochs):
        # Important: makes sure shuffling works across epochs
        if train_sampler:
            train_sampler.set_epoch(epoch)

        next_batch = tuple(
            v.to(device, non_blocking=True) for v in next(iter(train_dataloader))
        )

        # Iterable dataset has no len()
        steps = (
            len(train_dataloader)
            if not isinstance(train_dataloader.dataset, IterableDataset)
            else trainer_config.max_iters
        )
        print(
            f"[rank {local_rank}] epoch {epoch} | bsz: {trainer_config.train_batch_size} | steps: {steps}"
        )

        with profile_context as prof:
            for step, batch_ahead in enumerate(train_dataloader):
                if trainer_config.max_iters and step == trainer_config.max_iters:
                    break

                # Record batch on current stream properly so the underlying memory isn't freed
                _wait_batch(next_batch, memcpy_stream)

                batch = next_batch
                images, captions, targets = batch

                # Prefetch and copy data for next batch
                with torch.cuda.stream(memcpy_stream):
                    next_batch = tuple(
                        v.to(device, non_blocking=True) for v in batch_ahead
                    )

                t0 = time.time()

                if trainer_config.cuda_graph:
                    # compiled graph contains model forward, backward and optimizer step
                    logits, loss = compiled_model(images, captions, targets)
                else:
                    with mp_context:
                        logits, loss = model(images, captions, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)

                if trainer_config.log_verbose:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    t1 = time.time()

                    print(
                        f"[rank {local_rank}] epoch {epoch}, step {step}, dt: {1000.0*(t1-t0):.4f}ms"
                    )

                if master_process:
                    if trainer_config.use_wandb:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "global_step": global_step,
                            }
                        )
                    if step % trainer_config.eval_interval == 0:
                        print(f"Loss at iteration {step}: {loss.item()}")

                if should_profile and master_process:
                    # for profiler scheduler
                    prof.step()

                global_step += 1

        if should_profile and master_process:
            export_trace(
                trainer_config.profile_results_dir, trainer_config.profiler_mode, prof
            )

        if master_process:
            run_validation(
                model,
                mp_context,
                epoch,
                global_step,
                trainer_config,
                val_dataloader,
                device,
            )
            save_snapshot(model, optimizer, scaler, trainer_config, epoch, wandb_run_id)


def get_model(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    vocab_size: int,
    device: torch.device,
    local_rank: int,
):
    model = VisionLanguageModel(
        model_config.n_embd,
        model_config.image_embed_dim,
        vocab_size,
        model_config.n_layer,
        model_config.img_size,
        model_config.patch_size,
        model_config.n_head,
        model_config.num_blks,
        model_config.emb_dropout,
        model_config.blk_dropout,
    )
    model = model.to(device)

    if trainer_config.use_ddp and not trainer_config.cuda_graph:
        # if we are using CUDA graph with DDP, defer DDP wrapping to compile_cuda_graph()
        model = DDP(model, device_ids=[local_rank])

    if trainer_config.compile:
        print("Using torch.compile(fullgraph=True, mode='default')")
        model = torch.compile(model, fullgraph=True)

    return model


@hydra.main(version_base=None, config_path=".", config_name="coco_train")
def main(config: DictConfig):
    trainer_config = TrainerConfig(**config["trainer_config"])
    use_ddp = int(os.environ.get("RANK", -1)) != -1
    trainer_config.use_ddp = use_ddp
    if not torch.cuda.is_available() and trainer_config.cuda_graph:
        raise Exception("cuda_graph=True set but no CUDA device available")

    model_config = ModelConfig(**config["model_config"])
    data_config = DataConfig(**config["data_config"])

    local_rank = 0
    # Env variables here are set by torchrun
    if trainer_config.use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        # To enable CUDA graph with DDP
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # NOTE: I had to disable this on 2 x RTX 4000 Ada DDP run
        # otherwise, GPUs would try to use P2P and fail to communicate
        # in my case, the GPUs were connected by NVLink
        os.environ["NCCL_P2P_DISABLE"] = "1"
        # os.environ['NCCL_P2P_LEVEL'] = 'NVL'

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        print(f"Rank: {local_rank} starting training")

    if local_rank == 0:
        print(f"Model config loaded:\n{model_config}\n")
        print(f"Trainer config loaded:\n{trainer_config}\n")
        print(f"Data config loaded:\n{data_config}\n")
        print("=" * 15)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Enable cudNN auto-tuner
    torch.backends.cudnn.benchmark = True

    # Use TF32 for speed-up where possible
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start = time.time()

    train_model(model_config, trainer_config, data_config)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    print(f"[rank {local_rank}] Training time took {elapsed:.2f} seconds")

    if trainer_config.use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
