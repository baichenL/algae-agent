import datetime
import os
import sqlite3

from app.core.db.connection import DB_PATH
from app.core.time_utils import local_time_string
from app.core.db.scientific import init_scientific_schema
from app.core.db.control_plane import init_control_plane_schema


def init_db():
    """初始化数据库，严格记录实验室物理存在的藻株及其详尽的中英文元数据"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # 注意：不再在初始化时删除已有的 `algae_status` 表，以免重启覆盖生产数据。
        # 仅在表不存在时创建（保留历史数据与手动写入记录）。
        # 创建核心状态表，增加中英文名称字段
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS algae_status (
                strain_id TEXT PRIMARY KEY,
                name_cn TEXT NOT NULL,
                name_en TEXT NOT NULL,
                generation_number INTEGER NOT NULL,
                days_since_last_subculture INTEGER NOT NULL,
                last_subculture_time TEXT
            )
        """)
        
        current_time = local_time_string()
        
        # 🌟 显式注入一号藻种（莱茵衣藻）和二号藻种（小球藻）的完整信息
        cursor.execute("""
            INSERT OR IGNORE INTO algae_status 
            (strain_id, name_cn, name_en, generation_number, days_since_last_subculture, last_subculture_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Chlamydomonas_01", "莱茵衣藻", "Chlamydomonas reinhardtii", 1, 0, current_time))
        
        cursor.execute("""
            INSERT OR IGNORE INTO algae_status 
            (strain_id, name_cn, name_en, generation_number, days_since_last_subculture, last_subculture_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Chlorella_01", "小球藻", "Chlorella vulgaris", 14, 0, current_time))
        
        conn.commit()
        init_scientific_schema(cursor)
        init_control_plane_schema(cursor)
        conn.commit()
        # 创建实验历史表 Level 2: experiments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strain TEXT NOT NULL,
                generation INTEGER,
                status TEXT,
                media_components_json TEXT,
                temperature REAL,
                light_intensity REAL,
                od_readings_json TEXT,
                hardware_logs TEXT,
                created_at TEXT,
                reflected INTEGER DEFAULT 0
            )
        """)

        # 创建反思规则表 Level 3: reflection_rules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflection_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                effect_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_experiments_json TEXT NOT NULL,
                notes TEXT,
                created_at TEXT
            )
        """)

        conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                source_run_id TEXT,
                confidence REAL NOT NULL DEFAULT 0.5,
                last_used_at TEXT,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(scope, content)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_memories_scope_status
            ON agent_memories (scope, status, confidence)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                trigger_patterns_json TEXT NOT NULL,
                procedure_markdown TEXT NOT NULL,
                risk_boundary TEXT NOT NULL,
                examples_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                version INTEGER NOT NULL DEFAULT 1,
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_skills_status_usage
            ON agent_skills (status, usage_count, success_count, failure_count)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_learning_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_run_id TEXT,
                session_id TEXT,
                review_type TEXT NOT NULL,
                candidate_type TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                run_digest_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending_review',
                validation_reason TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_learning_reviews_run
            ON agent_learning_reviews (agent_run_id, id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_learning_reviews_status
            ON agent_learning_reviews (status, id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_learning_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_run_id TEXT,
                session_id TEXT,
                artifact_type TEXT NOT NULL,
                artifact_id INTEGER NOT NULL,
                usage_context TEXT,
                outcome TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_learning_usage_artifact
            ON agent_learning_usage (artifact_type, artifact_id, id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memory_settings (
                owner_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0, 1)),
                auto_write_low_risk INTEGER NOT NULL DEFAULT 1 CHECK(auto_write_low_risk IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(owner_id, workspace_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('preference', 'profile', 'note')),
                subject TEXT NOT NULL DEFAULT 'user',
                predicate TEXT NOT NULL,
                value_json TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                search_text TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
                sensitivity TEXT NOT NULL DEFAULT 'low' CHECK(sensitivity IN ('low', 'sensitive', 'prohibited')),
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived')),
                revision INTEGER NOT NULL DEFAULT 1,
                valid_from TEXT,
                valid_to TEXT,
                source_conversation_id TEXT,
                source_message_id INTEGER,
                source_run_id TEXT,
                use_count INTEGER NOT NULL DEFAULT 0,
                confirmation_count INTEGER NOT NULL DEFAULT 0,
                correction_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_memories_owner_status
            ON user_memories (owner_id, workspace_id, status, predicate, updated_at)
        """)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memories_singleton_active
            ON user_memories (owner_id, workspace_id, subject, predicate)
            WHERE status = 'active' AND predicate != 'general.note'
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memory_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('preference', 'profile', 'note')),
                subject TEXT NOT NULL DEFAULT 'user',
                predicate TEXT NOT NULL,
                value_json TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                normalized_hash TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
                sensitivity TEXT NOT NULL DEFAULT 'low' CHECK(sensitivity IN ('low', 'sensitive', 'prohibited')),
                reason TEXT,
                evidence TEXT,
                model_name TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected', 'blocked')),
                source_conversation_id TEXT,
                source_message_id INTEGER,
                source_run_id TEXT,
                applied_memory_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_id, workspace_id, source_message_id, predicate, normalized_hash)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_memory_candidates_owner
            ON user_memory_candidates (owner_id, workspace_id, status, id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                candidate_id INTEGER,
                owner_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT,
                reason TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_memory_events_owner
            ON user_memory_events (owner_id, workspace_id, id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memory_embeddings (
                memory_id INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(memory_id, embedding_model)
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS user_memories_fts
            USING fts5(
                memory_id UNINDEXED,
                owner_id UNINDEXED,
                workspace_id UNINDEXED,
                predicate,
                search_text,
                tokenize='unicode61'
            )
        """)

        conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_error_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_run_id TEXT,
                layer TEXT NOT NULL,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'error',
                error_type TEXT,
                error_message TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_error_events_run
            ON agent_error_events (agent_run_id, id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_error_events_layer
            ON agent_error_events (layer, component, id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_message TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT,
                finished_at TEXT,
                final_route TEXT,
                risk_level TEXT,
                response_summary TEXT,
                error_event_id INTEGER,
                FOREIGN KEY(error_event_id) REFERENCES agent_error_events(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_runs_session
            ON agent_runs (session_id, started_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_runs_status
            ON agent_runs (status, started_at)
        """)
        _ensure_agent_run_columns(cursor)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assistant_conversations (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT 'shared',
                title TEXT NOT NULL DEFAULT '新对话',
                status TEXT NOT NULL DEFAULT 'active',
                legacy_session_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_message_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assistant_conversations_owner
            ON assistant_conversations (owner, workspace_id, status, updated_at)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assistant_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                run_id TEXT,
                structured_json TEXT,
                client_message_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES assistant_conversations(id),
                UNIQUE(conversation_id, client_message_id)
            )
        """)
        _ensure_assistant_message_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation
            ON assistant_messages (conversation_id, id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT 'shared',
                task_type TEXT NOT NULL DEFAULT 'conversation_turn',
                goal_text TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                state_changing INTEGER NOT NULL DEFAULT 0,
                collected_slots_json TEXT,
                missing_slots_json TEXT,
                proposed_action_json TEXT,
                pending_id INTEGER,
                workflow_run_id TEXT,
                latest_agent_run_id TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                legacy_source TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                completed_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_tasks_conversation
            ON conversation_tasks (conversation_id, updated_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_tasks_owner
            ON conversation_tasks (owner, workspace_id, status, updated_at)
        """)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_tasks_one_active_change
            ON conversation_tasks (conversation_id)
            WHERE state_changing = 1
              AND status IN ('collecting', 'ready', 'waiting_approval', 'running', 'waiting_manual_confirmation')
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(task_id, sequence),
                FOREIGN KEY(task_id) REFERENCES conversation_tasks(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_task_events_resume
            ON conversation_task_events (task_id, sequence)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS background_operations (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT 'shared',
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                phase TEXT NOT NULL DEFAULT 'queued',
                message TEXT,
                progress REAL,
                related_run_id TEXT,
                related_entity_type TEXT,
                related_entity_id TEXT,
                retryable INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                heartbeat_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        _ensure_background_operation_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_background_operations_owner
            ON background_operations (owner, workspace_id, status, updated_at)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                phase TEXT,
                message TEXT,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(operation_id, sequence),
                FOREIGN KEY(operation_id) REFERENCES background_operations(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_operation_events_resume
            ON operation_events (operation_id, sequence)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_runs_graph_thread
            ON agent_runs (graph_thread_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_runs_task
            ON agent_runs (task_id, started_at)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_run_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT,
                FOREIGN KEY(agent_run_id) REFERENCES agent_runs(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_run_events_run
            ON agent_run_events (agent_run_id, id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_run_events_type
            ON agent_run_events (event_type, id)
        """)

        conn.commit()

        # 创建审计日志表，用于记录对 algae_status 的增删改操作
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS algae_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strain_id TEXT,
                action TEXT,
                details_json TEXT,
                performed_at TEXT
            )
        """)

        conn.commit()

        # 待人工确认的操作（由 LLM 发起但需人工确认的修改请求）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT,
                requester TEXT,
                status TEXT DEFAULT 'pending',
                reviewed_at TEXT,
                reviewed_by TEXT,
                review_reason TEXT,
                risk_level TEXT DEFAULT 'medium',
                source TEXT DEFAULT 'chat'
            )
        """)
        _ensure_pending_action_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_actions_graph_thread
            ON pending_actions (graph_thread_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_actions_agent_run
            ON pending_actions (agent_run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_actions_execution_key
            ON pending_actions (execution_idempotency_key)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_actions_domain_key
            ON pending_actions (domain_dedupe_key)
        """)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_actions_execution_key
            ON pending_actions (execution_idempotency_key)
            WHERE execution_idempotency_key IS NOT NULL
        """)
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_actions_active_domain_key
            ON pending_actions (domain_dedupe_key)
            WHERE domain_dedupe_key IS NOT NULL AND status = 'pending'
        """)

        conn.commit()

        # 创建 workflow_runs 表，用于记录每次 workflow 的执行情况
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pending_id INTEGER NOT NULL UNIQUE,
                workflow_name TEXT NOT NULL,
                strain_id TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                expected_generation INTEGER,
                tool_result_json TEXT,
                physical_execution INTEGER NOT NULL DEFAULT 0,
                persisted INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                FOREIGN KEY(pending_id) REFERENCES pending_actions(id)
            )
        """)
        _ensure_workflow_run_columns(cursor)
        # 创建索引以加速查询
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_status
            ON workflow_runs (status, id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_runs_strain
            ON workflow_runs (strain_id, id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approval_resume_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pending_id INTEGER NOT NULL,
                agent_run_id TEXT,
                graph_thread_id TEXT,
                approval_version INTEGER,
                decision TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(pending_id, approval_version)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_resume_jobs_status
            ON approval_resume_jobs (status, attempt_count, id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_resume_jobs_thread
            ON approval_resume_jobs (graph_thread_id)
        """)

        conn.commit()
        # 创建 subculture_reminder_cycles 和 subculture_reminder_events 表，用于管理传代提醒周期和事件
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subculture_reminder_cycles (
                cycle_key TEXT PRIMARY KEY,
                strain_id TEXT NOT NULL,
                generation_number INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                first_due_at TEXT,
                first_sent_at TEXT,
                last_sent_at TEXT,
                total_sent_count INTEGER NOT NULL DEFAULT 0,
                silenced_at TEXT,
                silenced_reason TEXT,
                resolved_at TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subculture_reminder_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_key TEXT NOT NULL,
                window_key TEXT NOT NULL,
                sent_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT,
                email_log_id TEXT,
                reason TEXT,
                created_at TEXT,
                metadata_json TEXT,
                UNIQUE(cycle_key, window_key)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subculture_reminder_events_cycle_date
            ON subculture_reminder_events (cycle_key, sent_date, event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_subculture_reminder_cycles_strain_generation
            ON subculture_reminder_cycles (strain_id, generation_number)
        """)

        conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                topic TEXT,
                version TEXT,
                year TEXT,
                language TEXT,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                indexed_at TEXT,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL UNIQUE,
                doc_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                title TEXT,
                section TEXT,
                page_number INTEGER,
                sheet_name TEXT,
                row_start INTEGER,
                row_end INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
            ON rag_chunks (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc_type
            ON rag_chunks (doc_type)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunk_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                document_id INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                vector_backend TEXT NOT NULL DEFAULT 'sqlite_blob_fallback',
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_chunk_embeddings_document
            ON rag_chunk_embeddings (document_id, embedding_model)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_chunk_embeddings_hash
            ON rag_chunk_embeddings (content_hash, embedding_model)
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                doc_type UNINDEXED,
                file_name,
                title,
                section,
                content,
                tokenize='unicode61'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT,
                citations_json TEXT,
                uncertainty_json TEXT,
                blocked_reason TEXT,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_source_schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                sheet_name TEXT,
                column_name TEXT NOT NULL,
                normalized_column_name TEXT,
                unit TEXT,
                inferred_role TEXT,
                column_index INTEGER,
                sample_values_json TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_source_schemas_document_id
            ON rag_source_schemas (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_source_schemas_source_file
            ON rag_source_schemas (source_file)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_recipe_components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                entity TEXT,
                group_name TEXT,
                component_name TEXT NOT NULL,
                normalized_component_name TEXT,
                amount REAL,
                unit TEXT,
                amount_text TEXT,
                solution_type TEXT,
                working_addition TEXT,
                section TEXT,
                table_index INTEGER,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_recipe_components_document_id
            ON rag_recipe_components (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_recipe_components_component
            ON rag_recipe_components (normalized_component_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_recipe_components_group
            ON rag_recipe_components (group_name)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_sop_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                entity TEXT,
                attribute TEXT,
                value TEXT,
                text_span TEXT,
                section TEXT,
                page_number INTEGER,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_sop_facts_document_id
            ON rag_sop_facts (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_sop_facts_entity
            ON rag_sop_facts (entity)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_paper_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                paper_title TEXT,
                year TEXT,
                entity TEXT,
                attribute TEXT,
                value TEXT,
                text_span TEXT,
                page_number INTEGER,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_paper_facts_document_id
            ON rag_paper_facts (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_paper_facts_attribute
            ON rag_paper_facts (attribute)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_experiment_data_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT NOT NULL,
                sheet_name TEXT,
                row_index INTEGER,
                column_name TEXT NOT NULL,
                normalized_column_name TEXT,
                inferred_role TEXT,
                value_text TEXT,
                numeric_value REAL,
                unit TEXT,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_experiment_values_document_id
            ON rag_experiment_data_values (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_experiment_values_role
            ON rag_experiment_data_values (inferred_role)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_experiment_values_row
            ON rag_experiment_data_values (source_file, sheet_name, row_index)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_knowledge_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL DEFAULT 'local_file',
                doc_type TEXT NOT NULL,
                source_path TEXT NOT NULL UNIQUE,
                file_name TEXT,
                version TEXT,
                year TEXT,
                language TEXT,
                trust_level TEXT NOT NULL DEFAULT 'lab_internal',
                owner TEXT,
                ingestion_status TEXT NOT NULL DEFAULT 'pending',
                content_hash TEXT,
                metadata_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        _ensure_rag_knowledge_source_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_knowledge_sources_doc_type
            ON rag_knowledge_sources (doc_type, trust_level)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_evidence_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                document_version TEXT,
                source_id TEXT,
                evidence_id TEXT NOT NULL UNIQUE,
                evidence_type TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                source_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                entity TEXT,
                relation TEXT,
                target_entity TEXT,
                attribute TEXT,
                value TEXT,
                unit TEXT,
                text_span TEXT,
                page_number INTEGER,
                section TEXT,
                section_path TEXT,
                sheet_name TEXT,
                row_start INTEGER,
                row_end INTEGER,
                parent_id TEXT,
                previous_id TEXT,
                next_id TEXT,
                source_locator TEXT,
                parser_version TEXT,
                element_type TEXT,
                confidence REAL,
                extraction_method TEXT,
                location_json TEXT,
                citation_json TEXT,
                metadata_json TEXT,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES rag_documents(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_source
            ON rag_evidence_units (source_id, evidence_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_entity
            ON rag_evidence_units (entity, relation, target_entity)
        """)
        _ensure_rag_evidence_unit_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_document
            ON rag_evidence_units (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_content_hash
            ON rag_evidence_units (content_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_parent
            ON rag_evidence_units (parent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_units_element_type
            ON rag_evidence_units (element_type)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_document_elements (
                element_id TEXT PRIMARY KEY,
                document_id TEXT,
                document_version TEXT,
                element_type TEXT NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                section_path TEXT,
                parent_id TEXT,
                previous_id TEXT,
                next_id TEXT,
                source_locator TEXT,
                metadata_json TEXT,
                content_hash TEXT NOT NULL,
                parser_name TEXT,
                parser_version TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_document_elements_document
            ON rag_document_elements (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_document_elements_parent
            ON rag_document_elements (parent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_document_elements_type
            ON rag_document_elements (element_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_document_elements_hash
            ON rag_document_elements (content_hash)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_trace_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL UNIQUE,
                user_query TEXT NOT NULL,
                route TEXT,
                query_frame_json TEXT,
                retrieval_channels_json TEXT,
                retrieved_ids_json TEXT,
                reranked_ids_json TEXT,
                selected_evidence_ids_json TEXT,
                answerability_json TEXT,
                sufficiency_json TEXT,
                citations_json TEXT,
                latency_ms INTEGER,
                status TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_retrieval_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                evidence_id TEXT,
                document_id TEXT,
                rank INTEGER,
                sparse_score REAL,
                dense_score REAL,
                fusion_score REAL,
                rerank_score REAL,
                selected INTEGER DEFAULT 0,
                rejection_reason TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_retrieval_traces_query
            ON rag_retrieval_traces (query_id, stage, rank)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_eval_cases (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                question TEXT NOT NULL,
                expected_route TEXT,
                required_evidence_types_json TEXT,
                expected_answer_contains_json TEXT,
                expected_refusal INTEGER DEFAULT 0,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases_json TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                evidence_id TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_relations_lookup
            ON rag_relations (source_entity_id, relation, target_entity_id)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_evidence_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                role TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_evidence_links_evidence
            ON rag_evidence_links (evidence_id, entity_id)
        """)

        conn.commit()


def _ensure_pending_action_columns(cursor):
    cursor.execute("PRAGMA table_info(pending_actions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "status": "TEXT DEFAULT 'pending'",
        "reviewed_at": "TEXT",
        "reviewed_by": "TEXT",
        "review_reason": "TEXT",
        "risk_level": "TEXT DEFAULT 'medium'",
        "source": "TEXT DEFAULT 'chat'",
        "agent_run_id": "TEXT",
        "graph_thread_id": "TEXT",
        "approval_version": "INTEGER DEFAULT 0",
        "execution_idempotency_key": "TEXT",
        "domain_dedupe_key": "TEXT",
        "expires_at": "TEXT",
        "executed_at": "TEXT",
        "execution_result_json": "TEXT",
        "resume_requested_at": "TEXT",
        "resume_completed_at": "TEXT",
        "resume_error": "TEXT",
        "execution_status": "TEXT DEFAULT 'not_started'",
        "execution_started_at": "TEXT",
        "execution_error": "TEXT",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE pending_actions ADD COLUMN {column_name} {column_def}")


def _ensure_workflow_run_columns(cursor):
    cursor.execute("PRAGMA table_info(workflow_runs)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "commit_source": "TEXT",
        "manual_confirmed_at": "TEXT",
        "manual_confirmed_by": "TEXT",
        "manual_note": "TEXT",
        "manual_commit_key": "TEXT",
        "current_step": "TEXT",
        "progress": "REAL DEFAULT 0",
        "hardware_state_json": "TEXT",
        "last_event_at": "TEXT",
        "last_event_sequence": "INTEGER DEFAULT 0",
        "updated_at": "TEXT",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE workflow_runs ADD COLUMN {column_name} {column_def}")


def _ensure_assistant_message_columns(cursor):
    cursor.execute("PRAGMA table_info(assistant_messages)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "status": "TEXT DEFAULT 'completed'",
        "operation_id": "TEXT",
        "task_id": "TEXT",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE assistant_messages ADD COLUMN {column_name} {column_def}")


def _ensure_rag_knowledge_source_columns(cursor):
    cursor.execute("PRAGMA table_info(rag_knowledge_sources)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "lifecycle_status": "TEXT DEFAULT 'active'",
        "archived_at": "TEXT",
        "archived_by": "TEXT",
        "chunk_count": "INTEGER DEFAULT 0",
        "last_error": "TEXT",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE rag_knowledge_sources ADD COLUMN {column_name} {column_def}")


def _ensure_agent_run_columns(cursor):
    cursor.execute("PRAGMA table_info(agent_runs)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "graph_thread_id": "TEXT",
        "checkpoint_status": "TEXT",
        "interrupted_at": "TEXT",
        "resumed_at": "TEXT",
        "resume_count": "INTEGER DEFAULT 0",
        "pending_id": "INTEGER",
        "last_node": "TEXT",
        "graph_definition_version": "TEXT",
        "state_schema_version": "INTEGER",
        "conversation_id": "TEXT",
        "task_id": "TEXT",
        "attempt_no": "INTEGER DEFAULT 1",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE agent_runs ADD COLUMN {column_name} {column_def}")


def _ensure_background_operation_columns(cursor):
    cursor.execute("PRAGMA table_info(background_operations)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "related_task_id" not in existing_columns:
        cursor.execute("ALTER TABLE background_operations ADD COLUMN related_task_id TEXT")


def _ensure_rag_evidence_unit_columns(cursor):
    cursor.execute("PRAGMA table_info(rag_evidence_units)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    required_columns = {
        "document_version": "TEXT",
        "content": "TEXT",
        "content_hash": "TEXT",
        "section_path": "TEXT",
        "parent_id": "TEXT",
        "previous_id": "TEXT",
        "next_id": "TEXT",
        "source_locator": "TEXT",
        "parser_version": "TEXT",
        "element_type": "TEXT",
    }
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE rag_evidence_units ADD COLUMN {column_name} {column_def}")
