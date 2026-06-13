import asyncio
import time
from typing import List, Dict, Any, Callable, Tuple, Optional

class DynamicBatcher:
    def __init__(self, processor_func: Callable[[List[Any]], List[Any]], max_batch_size: int = 16, max_wait_ms: float = 50.0):
        """
        Args:
            processor_func: The batch-processing worker function that accepts a list of inputs and returns a list of outputs.
            max_batch_size: Maximum items to package in one execution.
            max_wait_ms: Maximum time to wait for a full batch before forcing execution.
        """
        self.processor_func = processor_func
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False

    def start(self):
        self.running = True
        self.worker_task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, item: Any) -> Any:
        """
        Enqueues an item and waits for its specific result from the batch processor.
        """
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((item, future))
        return await future

    async def _batch_loop(self):
        while self.running:
            # Wait for at least one item to arrive
            item, future = await self.queue.get()
            batch = [(item, future)]
            
            start_time = time.time()
            
            # Continue accumulating items until either batch is full or wait time is reached
            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000.0
                time_to_wait = (self.max_wait_ms - elapsed_ms) / 1000.0
                
                if time_to_wait <= 0:
                    break
                    
                try:
                    # Non-blocking get from queue with timeout
                    next_item, next_future = await asyncio.wait_for(self.queue.get(), timeout=max(0.001, time_to_wait))
                    batch.append((next_item, next_future))
                except asyncio.TimeoutError:
                    break
                    
            # Extract inputs and execute batch
            inputs = [b[0] for b in batch]
            try:
                # Execute in executor to avoid blocking the event loop for CPU/GPU heavy ops
                loop = asyncio.get_event_loop()
                # Run the processor function
                outputs = await loop.run_in_executor(None, self.processor_func, inputs)
                
                # Resolve futures
                for (_, fut), out in zip(batch, outputs):
                    if not fut.done():
                        fut.set_result(out)
            except Exception as e:
                # Set exception on all futures in the batch if it fails
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                for _ in range(len(batch) - 1): # We already did one queue.get() outside the loop
                    self.queue.task_done()
                self.queue.task_done()
