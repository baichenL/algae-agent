export type RunStatus = 'queued' | 'running' | 'waiting_input' | 'waiting_approval' | 'ready_to_execute' | 'succeeded' | 'failed' | 'blocked' | 'cancelled'

export interface RunSummary {
  id: string
  kind: string
  title: string
  workspace_id: string
  status: RunStatus
  phase: string
  progress: number
  cycle: number
  risk_level: string
  simulation_only: boolean
  source_ref: { type: string; id: string | number }
  created_at?: string
  updated_at?: string
  summary?: string
}

export interface Phase { key: string; label: string; state: 'completed' | 'active' | 'pending' | 'blocked' }
export interface ActionDescriptor { id: string; label: string; kind: string; approval_id?: number; task_id?: string; required_role: string; requires_confirmation: boolean }
export interface Approval {
  id: number; run_id?: string; action_type: string; status: string; risk_level: string
  design_hash?: string; approval_version: number; requested_by?: string; reviewed_by?: string
  review_reason?: string; created_at?: string; reviewed_at?: string; summary: string
  payload?: Record<string, unknown>; executed_at?: string
  execution_status?: string; execution_error?: string
}

export interface Conversation {
  id: string; owner: string; title: string; status: 'active' | 'archived'
  message_count?: number; updated_at?: string; last_message_at?: string
}

export interface ConversationMessage {
  id: number; role: 'user' | 'assistant'; content: string; run_id?: string
  structured?: unknown; client_message_id?: string; created_at?: string
  status?: 'processing' | 'completed' | 'failed'; operation_id?: string; task_id?: string
}
export interface AnswerEnvelope {
  schema_version: 'answer-envelope/v1' | 'answer-envelope/v2'
  outcome_status: 'success' | 'partial' | 'needs_input' | 'pending' | 'paused' | 'failed'
  direct_answer: string
  confirmed_facts: unknown[]
  inferences: unknown[]
  simulation_results: unknown[]
  recommendations: unknown[]
  unknowns: string[]
  source_statuses: Array<{ source: string; status: string; used?: boolean; impact?: string }>
  completed_work: Record<string, unknown>
  remaining_work: unknown[]
  budget: Record<string, unknown>
  references: Array<{ type: string; id: string | number }>
  next_actions: Array<{ action: string; label: string; task_id?: string; pending_id?: number }>
  presentation_blocks?: PresentationBlock[]
}
export type PresentationBlock =
  | { type: 'email_draft'; draft: Record<string, any>; target?: string; send: false }
  | { type: 'approval'; pending_id: number; status: string; executable: false; requires_human_approval: true }
  | { type: 'scientific_result'; candidate_count: number; candidates: unknown[]; validations: unknown[]; simulations: unknown[]; plan_patches: unknown[]; comparison: Record<string, unknown>; scientific_run_id?: string }
  | { type: 'paused_task'; task_id?: string; reason?: string; completed_work: Record<string, unknown>; remaining_work: unknown[]; budget: Record<string, unknown> }
export interface ConversationTask {
  id: string; conversation_id: string; task_type: string; goal_text: string
  status: 'collecting' | 'ready' | 'waiting_approval' | 'running' | 'waiting_manual_confirmation' | 'suspended' | 'paused' | 'completed' | 'cancelled' | 'failed' | 'expired'
  collected_slots: Record<string, unknown>; missing_slots: string[]
  proposed_action?: Record<string, any>; pending_id?: number; workflow_run_id?: string
  latest_agent_run_id?: string; version: number; updated_at?: string
}
export interface OperationSummary {
  id: string; kind: string; label: string
  status: 'queued' | 'running' | 'waiting_input' | 'succeeded' | 'failed' | 'cancelled'
  phase: string; message?: string; progress?: number; related_run_id?: string
  retryable: boolean; error_message?: string; created_at: string; started_at?: string
  heartbeat_at?: string; completed_at?: string; updated_at: string
}
export interface OperationEvent {
  sequence: number; event_type: string; phase?: string; message?: string
  payload: Record<string, unknown>; created_at: string
}
export interface ReplanSummary {
  id: string; kind: 'agent' | 'scientific'; trigger_observation: Record<string, any>
  original_plan: Record<string, any>; reason: string; revised_plan: Record<string, any>
  key_changes: string[]; from_version: number; to_version: number
  final_action?: string; occurred_at?: string; safety_boundary?: string
  debug?: Record<string, any>
}
export interface MotionPhase {
  name: string
  duration_ms: number
  easing: 'minimum_jerk' | 's_curve' | 'linear_cruise' | 'hold' | string
  pose_from?: string
  pose_to?: string
  path?: string[]
}
export interface AttachmentTransition {
  entity_id: string
  phase: string
  from_parent?: string | null
  to_parent: string
}
export interface MotionCommand {
  schema_version: 'motion-v2' | string
  command_id: string
  actor_id: string
  entity_id?: string | null
  action_type: string
  simulation_start_ms: number
  nominal_duration_ms: number
  required_resources: string[]
  preconditions: string[]
  postconditions: string[]
  phases: MotionPhase[]
  attachment_transition?: AttachmentTransition
  source_location?: string | null
  target_location?: string | null
  interruptibility?: string
  safe_stop_phase?: string
}
export interface SimulationEntity {
  entity_type: string
  location?: string
  parent_id?: string
  [key: string]: unknown
}
export interface SimulationSnapshot {
  phase?: string; uv?: Record<string, any>; media_reservoir?: Record<string, any>
  source_reactor?: Record<string, any>; target_reactor?: Record<string, any>
  media_pump?: Record<string, any>; seed_pump?: Record<string, any>
  valve?: Record<string, any>; incubator?: Record<string, any>
  spectrophotometer?: Record<string, any>; liquid_handler?: Record<string, any>
  plate_reader?: Record<string, any>; measurement_plate?: Record<string, any>
  robot_arm?: Record<string, any>; mobile_robot?: Record<string, any>
  transport?: {
    carrier?: string | null; labware_id?: string | null
    source_location?: string | null; target_location?: string | null
    motion_phase?: string; progress?: number
  }
  motion_schema_version?: string
  simulation_time_ms?: number
  entities?: Record<string, SimulationEntity>
  attachments?: Record<string, string>
  resource_occupancy?: Record<string, string>
  active_commands?: MotionCommand[]
  device_interlocks?: Record<string, string>
  sample?: Record<string, any>; alarm?: Record<string, any> | null
}
export interface SimulationEvent {
  sequence: number; event_type: string; phase?: string; level: string; created_at?: string
  payload: {
    step?: string; device?: string; action?: string; status?: string
    task_id?: string
    step_progress?: number; progress?: number; message?: string
    motion?: SimulationSnapshot['transport']
    motion_command?: MotionCommand
    snapshot?: SimulationSnapshot; simulation_only?: boolean; error?: unknown
  }
}
export interface Artifact { id: number; artifact_type: string; version: number; payload: Record<string, any>; content_hash: string; created_at?: string }
export interface Assertion { name: string; status: string; details?: string }
export interface RunDetail extends RunSummary {
  phases: Phase[]
  current_action?: ActionDescriptor
  available_actions: ActionDescriptor[]
  artifacts: Artifact[]
  approvals: Approval[]
  executions: any[]
  assertions: Assertion[]
  replans?: ReplanSummary[]
  simulation?: {
    current_step?: string
    progress: number
    latest_snapshot?: SimulationSnapshot
    event_count: number
    replay_available: boolean
    last_event_at?: string
    queue_position?: number | null
    interaction_mode?: string
  }
  trace?: Record<string, any>
  raw_events?: any[]
  raw?: Record<string, any>
}

export interface SessionInfo {
  status: string
  csrf_token?: string
  principal: { name: string; role: string }
  workspace: { id: string; name: string; mode: string }
}

export interface UserMemory {
  id: number
  owner_id: string
  workspace_id: string
  memory_type: 'preference' | 'profile' | 'note'
  predicate: string
  value: unknown
  confidence: number
  sensitivity: string
  status: 'active' | 'superseded' | 'archived'
  revision: number
  source_conversation_id?: string
  source_message_id?: number
  use_count: number
  confirmation_count: number
  correction_count: number
  updated_at: string
}

export interface UserMemoryCandidate {
  id: number
  memory_type: 'preference' | 'profile' | 'note'
  predicate: string
  value: unknown
  confidence: number
  reason?: string
  evidence?: string
  status: 'pending' | 'accepted' | 'rejected' | 'blocked'
  source_conversation_id?: string
  created_at: string
}

export interface UserMemorySettings {
  owner_id: string
  workspace_id: string
  enabled: boolean
  auto_write_low_risk: boolean
}

export interface EvaluationMetric {
  key: string
  display_name: string
  value: number | null
  unit: 'ratio' | 'count' | 'milliseconds' | 'tokens' | 'regret'
  numerator?: number
  denominator?: number
  ci95?: [number, number]
  standard: string
  reference_url: string
  evidence_mode: 'deterministic' | 'llm_judge'
  status: 'completed' | 'not_run' | 'failed'
  status_reason?: string
  definition?: string
  project_target?: number
}

export interface EvaluationSuite {
  key: string
  display_name: string
  status: 'completed' | 'passed' | 'not_run' | 'failed'
  status_reason?: string
  description?: string
  sample_count: number
  metrics: EvaluationMetric[]
  details: Record<string, any>
}

export interface EvaluationReport {
  schema_version: '1.0'
  report_id: string
  generated_at: string
  git_commit: string
  dataset_versions: Record<string, string>
  environment: {
    embedding_backend: string
    reranker_backend: string
    judge_model?: string
    judge_enabled?: boolean
    embedding_used?: boolean
    reranker_used?: boolean
    judge_used?: boolean
  }
  suites: EvaluationSuite[]
  comparisons: Array<Record<string, any>>
}
