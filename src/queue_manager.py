import asyncio
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import io
from src.whisper_utils import whisper_transcribe
from src.config import QUEUE_WORKERS, MAX_QUEUE_SIZE


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, audio_data: bytes, filename: str, language: Optional[str] = None):
        self.job_id = job_id
        self.audio_data = audio_data
        self.filename = filename
        self.language = language
        self.status = JobStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None


class TranscriptionQueueManager:
    def __init__(self, num_workers: int = QUEUE_WORKERS, max_queue_size: int = MAX_QUEUE_SIZE):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.jobs: Dict[str, Job] = {}
        self.num_workers = num_workers
        self.workers = []
        self.is_running = False

    async def start(self):
        """Start the queue workers"""
        if self.is_running:
            return
        
        self.is_running = True
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        print(f"Started {self.num_workers} transcription workers")

    async def stop(self):
        """Stop all queue workers"""
        self.is_running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        print("Stopped all transcription workers")

    async def submit_job(self, audio_data: bytes, filename: str, language: Optional[str] = None) -> str:
        """Submit a new transcription job and return job ID"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, audio_data, filename, language)
        self.jobs[job_id] = job
        
        try:
            await self.queue.put(job)
        except asyncio.QueueFull:
            job.status = JobStatus.FAILED
            job.error = "Queue is full. Please try again later."
            raise Exception("Queue is full")
        
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        status_info = {
            "job_id": job.job_id,
            "status": job.status,
            "filename": job.filename,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
        
        if job.status == JobStatus.COMPLETED:
            status_info["result"] = job.result
        elif job.status == JobStatus.FAILED:
            status_info["error"] = job.error
        
        return status_info

    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about the queue"""
        pending_count = sum(1 for job in self.jobs.values() if job.status == JobStatus.PENDING)
        processing_count = sum(1 for job in self.jobs.values() if job.status == JobStatus.PROCESSING)
        completed_count = sum(1 for job in self.jobs.values() if job.status == JobStatus.COMPLETED)
        failed_count = sum(1 for job in self.jobs.values() if job.status == JobStatus.FAILED)
        
        return {
            "workers": self.num_workers,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.queue.maxsize,
            "total_jobs": len(self.jobs),
            "pending": pending_count,
            "processing": processing_count,
            "completed": completed_count,
            "failed": failed_count,
        }

    async def _worker(self, worker_id: int):
        """Worker that processes jobs from the queue"""
        print(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Update job status
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                print(f"Worker {worker_id} processing job {job.job_id}")
                
                try:
                    # Process the transcription in a thread pool to avoid blocking
                    audio_io = io.BytesIO(job.audio_data)
                    loop = asyncio.get_event_loop()
                    vtt_content, _ = await loop.run_in_executor(
                        None, 
                        whisper_transcribe,
                        audio_io,
                        job.language
                    )
                    
                    # Mark job as completed
                    job.status = JobStatus.COMPLETED
                    job.result = vtt_content
                    job.completed_at = datetime.now()
                    print(f"Worker {worker_id} completed job {job.job_id}")
                    
                except Exception as e:
                    # Mark job as failed
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now()
                    print(f"Worker {worker_id} failed job {job.job_id}: {str(e)}")
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.TimeoutError:
                # No job available, continue waiting
                continue
            except asyncio.CancelledError:
                # Worker is being stopped
                print(f"Worker {worker_id} stopped")
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {str(e)}")
                continue


# Global queue manager instance
_queue_manager: Optional[TranscriptionQueueManager] = None


def get_queue_manager() -> TranscriptionQueueManager:
    """Get or create the global queue manager instance"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = TranscriptionQueueManager()
    return _queue_manager

