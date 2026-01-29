## Speech to Text Engine: Hybrid Asynchronous Transcription

This module implements a real-time Speech-to-Text (STT) actor designed to minimize user-perceived latency. The system orchestrates voice activity detection (VAD) and transcription by maintaining a per-session state machine that intelligently offloads processing tasks to background workers.

1. Core Concept: The STTActor
The STTActor acts as a stateful orchestrator for a single audio stream session. Unlike traditional pipelines that wait for a full sentence to finish before processing, this actor employs a hybrid offloading strategy:

Base Model (Background): Handles longer, intermediate audio segments during natural pauses in speech.

Tail Model (Finalization): Handles the remaining audio "tail" immediately upon speech termination, typically using a faster, smaller model (e.g., Tiny).

2. Behavioral State Machine
The actor operates in three primary states to manage the audio stream efficiently:

stateDiagram-v2
    direction LR
    
    [*] --> Idle: Actor Ready
    
    state "Idle (Monitoring)" as Idle {
        [*] --> BufferAudio
        BufferAudio --> VADCheck: Process Chunk
        VADCheck --> BufferAudio: Silence
    }

    state "Recording Session" as Recording {
        state "Accumulating Audio" as Accumulating
        state "Offloading Background Task" as Offload
        
        [*] --> Accumulating
        
        Accumulating --> Accumulating: Continue Speech
        
        %% The core logic for background processing
        Accumulating --> Offload: Pause > 3s
        Offload --> Accumulating: Task Sent to 'Base' Model
    }

    state "Finalizing & Transcribing" as Finalizing {
        [*] --> CheckPending: Check Background Tasks
        CheckPending --> TranscribeTail: Send Remainder to 'Tail' Model
        TranscribeTail --> MergeResults: Gather All Text
        MergeResults --> [*]
    }

    %% Transitions
    Idle --> Recording: VAD 'Start' Detected
    Recording --> Finalizing: VAD 'End' Detected
    Finalizing --> Idle: Result Returned

A. Idle (Monitoring) The system buffers incoming raw audio chunks and processes them through a Silero VAD iterator (vad_controller). It remains in this state until a speech start event is detected.

B. Recording (The Offload Loop) Once speech begins, the system accumulates audio into a sentence_buffer. To prevent latency spikes at the end of long sentences, the system monitors for "pause" events using a secondary VAD iterator (vad_pause) with a lower threshold.

Trigger: If a pause is detected and the accumulated buffer exceeds the MIN_PIPELINE_DURATION (3.0 seconds).

Action: The buffered audio is flushed and sent asynchronously to the Base Whisper deployment via Ray Serve. The actor continues recording without blocking.

C. Finalizing (Aggregation) When the primary VAD detects the end of speech:

Language Check: The actor briefly checks if any background tasks have finished to extract a language_hint (e.g., "en", "de"), improving the accuracy of the final segment.

Tail Transcription: The remaining audio "tail" is sent to the Tail Whisper deployment (configured as a faster model).

Merge: The system awaits all pending futures (background + tail), concatenates the partial transcripts in order, and returns the final result.

3. Infrastructure
The system is built on Ray Serve, allowing the VAD logic (CPU-bound) to scale independently from the Whisper models (GPU/Compute-bound). The STTManager initializes two distinct deployment pools:

whsper_base_deployment: Optimized for accuracy on longer contexts.

whsper_tiny_deployment: Optimized for speed to finalize interaction quickly.