"""builderr · domain VOCABULARY BOOSTING module (standalone — touches no other module).

Maximizes recognition of AI / software / cloud / metrics / Indian-tech-Hinglish terms.

Exports:
    TECH_TERMS                 dict[str,str]   >= 1000 entries (lowercase -> canonical)
    PHRASE_HINTS               tuple[str]      >= 500 Hinglish tech phrases
    get_initial_prompt()       -> str          Whisper vocabulary-biasing prompt
    normalize_tech_words(text) -> str          'g p t four'->'GPT4', 'q lora'->'QLoRA', ...
    repair_common_asr_errors(text) -> str      'cursor sir'->'Cursor', 'aws p'->'AWS pe', ...

Pure standard library. No model loading, no network. Safe to import anywhere.
"""
from __future__ import annotations

import re

# =============================================================================
# 1. TECH_TERMS  (>= 1000 canonical entries)
# =============================================================================
TECH_TERMS: dict[str, str] = {}


def _add(*words: str) -> None:
    for w in words:
        k = w.lower()
        cur = TECH_TERMS.get(k)
        # keep the canonical form with the most uppercase (so "AWS" beats a later "aws"
        # added as a CLI command); genuine lowercase commands like "grep" are unaffected.
        if cur is None or sum(c.isupper() for c in w) > sum(c.isupper() for c in cur):
            TECH_TERMS[k] = w


# ---- AI / ML models, frameworks, concepts ----
_add(
    "GPT", "GPT2", "GPT3", "GPT3.5", "GPT4", "GPT4o", "GPT4.1", "GPT4.5", "GPT5", "ChatGPT",
    "Claude", "Claude2", "Claude3", "Claude3.5", "Claude3.7", "Claude4", "ClaudeOpus",
    "ClaudeSonnet", "ClaudeHaiku", "Opus", "Sonnet", "Haiku",
    "Gemini", "Gemini1.5", "Gemini2", "GeminiPro", "GeminiFlash", "GeminiUltra", "Bard", "PaLM",
    "Llama", "Llama2", "Llama3", "Llama3.1", "Llama3.2", "Llama4", "CodeLlama",
    "Qwen", "Qwen2", "Qwen2.5", "Qwen3", "QwenVL", "Qwen3-ASR",
    "Mistral", "Mixtral", "Mistral7B", "Codestral", "Ministral",
    "DeepSeek", "DeepSeekV2", "DeepSeekV3", "DeepSeekR1", "DeepSeekCoder",
    "Grok", "Phi", "Phi3", "Gemma", "Gemma2", "Falcon", "Vicuna", "Alpaca", "Yi", "Cohere",
    "Command", "CommandR", "Jamba", "Nemotron", "Granite", "StableLM", "Pythia", "MPT",
    "Whisper", "WhisperX", "Parakeet", "Nemo", "Canary", "Conformer", "Wav2Vec", "HuBERT",
    "SeamlessM4T", "Distil-Whisper", "FasterWhisper", "WhisperCpp", "VoskASR", "Vosk",
    "Sarvam", "IndicWhisper", "AI4Bharat", "Bhashini", "Srota",
    "RAG", "GraphRAG", "LoRA", "QLoRA", "DoRA", "PEFT", "RLHF", "DPO", "PPO", "SFT",
    "Embedding", "Embeddings", "Transformer", "Tokenizer", "Tokenization", "Attention",
    "SelfAttention", "CrossAttention", "FlashAttention", "Encoder", "Decoder", "Seq2Seq",
    "Diffusion", "StableDiffusion", "GAN", "VAE", "CNN", "RNN", "LSTM", "GRU", "MLP", "BERT",
    "RoBERTa", "DistilBERT", "ELECTRA", "T5", "BART", "XLNet", "ViT", "CLIP", "SAM", "YOLO",
    "ResNet", "EfficientNet", "MobileNet", "Inception", "AlexNet", "VGG", "UNet",
    "Quantization", "Distillation", "Pruning", "Finetuning", "Pretraining", "Inference",
    "Backpropagation", "Gradient", "Optimizer", "Adam", "AdamW", "SGD", "RMSProp", "Softmax",
    "ReLU", "GELU", "Sigmoid", "Tanh", "Dropout", "BatchNorm", "LayerNorm", "Epoch", "Batch",
    "Logits", "Perplexity", "Hallucination", "Prompt", "PromptEngineering", "FewShot",
    "ZeroShot", "ChainOfThought", "Agent", "Agentic", "MCP", "FunctionCalling", "ToolUse",
    "VectorDB", "VectorStore", "Reranker", "Retriever", "Chunking", "Context", "ContextWindow",
    "PyTorch", "TensorFlow", "Keras", "JAX", "Flax", "CUDA", "cuDNN", "ONNX", "TensorRT",
    "Triton", "vLLM", "HuggingFace", "Transformers", "Diffusers", "Datasets", "Accelerate",
    "DeepSpeed", "Megatron", "FSDP", "LangChain", "LangGraph", "LlamaIndex", "Haystack",
    "OpenAI", "Anthropic", "Mosaic", "Replicate", "Together", "Groq", "Fireworks", "Ollama",
    "LMStudio", "GGUF", "GGML", "Safetensors", "Tiktoken", "SentencePiece", "BPE",
    "scikit-learn", "XGBoost", "LightGBM", "CatBoost", "spaCy", "NLTK", "Gensim", "OpenCV",
)

# ---- programming languages ----
_add(
    "Python", "JavaScript", "TypeScript", "Java", "Kotlin", "Scala", "Groovy", "Clojure",
    "Golang", "Rust", "Swift", "ObjectiveC", "Ruby", "PHP", "Perl", "Lua", "Dart", "Julia",
    "Haskell", "Erlang", "Elixir", "FSharp", "OCaml", "Crystal", "Nim", "Zig", "Carbon",
    "Assembly", "Fortran", "COBOL", "Pascal", "Delphi", "Solidity", "Move", "Vyper",
    "Bash", "Zsh", "Fish", "PowerShell", "Batch", "Makefile", "CMake", "GraphQL", "WASM",
    "Verilog", "VHDL", "Prolog", "Scheme", "Racket", "Lisp", "Smalltalk", "Tcl", "Awk",
    "MATLAB", "Octave", "SAS", "Stata", "ABAP", "Apex", "PLSQL", "TSQL",
)

# ---- web / backend frameworks & libs ----
_add(
    "React", "ReactNative", "Angular", "AngularJS", "Vue", "Svelte", "SvelteKit", "Solid",
    "Preact", "Ember", "Backbone", "jQuery", "Redux", "MobX", "Zustand", "Recoil", "RxJS",
    "Next.js", "Nuxt", "Remix", "Gatsby", "Astro", "Vite", "Webpack", "Rollup", "Parcel",
    "Babel", "ESBuild", "SWC", "Turbopack", "ESLint", "Prettier", "Tailwind", "Bootstrap",
    "MaterialUI", "ChakraUI", "Bulma", "Sass", "Less", "PostCSS", "Storybook",
    "Node.js", "Deno", "Bun", "Express", "Koa", "Hapi", "NestJS", "Fastify", "Socket.io",
    "Django", "Flask", "FastAPI", "Tornado", "Sanic", "Starlette", "Pyramid", "Bottle",
    "Spring", "SpringBoot", "Hibernate", "Micronaut", "Quarkus", "Vertx", "Struts",
    "Rails", "Sinatra", "Laravel", "Symfony", "CodeIgniter", "CakePHP", "Phoenix", "Gin",
    "Echo", "Fiber", "Actix", "Rocket", "Axum", "Tokio", "Hyper", "Beego",
    "Pydantic", "SQLAlchemy", "Alembic", "Celery", "Pytest", "Unittest", "Tox", "Poetry",
    "Pip", "Conda", "Pipenv", "Virtualenv", "Numpy", "Pandas", "Scipy", "Matplotlib",
    "Seaborn", "Plotly", "Bokeh", "Dask", "Polars", "Requests", "HTTPX", "Aiohttp",
    "Boto3", "Pillow", "Streamlit", "Gradio", "Dash", "Jinja", "Pydeck",
    "JUnit", "Jest", "Mocha", "Chai", "Jasmine", "Karma", "Cypress", "Playwright",
    "Selenium", "Puppeteer", "Vitest", "Testify", "RSpec", "PHPUnit",
)

# ---- datastores / data engineering / streaming ----
_add(
    "Redis", "Memcached", "Kafka", "Pulsar", "RabbitMQ", "ActiveMQ", "ZeroMQ", "NATS",
    "Postgres", "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle", "SQLServer", "DB2",
    "MongoDB", "CouchDB", "Couchbase", "Cassandra", "ScyllaDB", "DynamoDB", "Bigtable",
    "HBase", "Neo4j", "ArangoDB", "OrientDB", "Dgraph", "JanusGraph", "CockroachDB",
    "YugabyteDB", "TiDB", "VoltDB", "Vitess", "PlanetScale", "Supabase", "Firebase",
    "Firestore", "Realm", "FaunaDB", "Elasticsearch", "OpenSearch", "Solr", "Lucene",
    "ClickHouse", "Druid", "Pinot", "InfluxDB", "TimescaleDB", "QuestDB", "Prometheus",
    "Snowflake", "Databricks", "Redshift", "BigQuery", "Synapse", "Firebolt", "Dremio",
    "Spark", "PySpark", "Hadoop", "HDFS", "Hive", "Presto", "Trino", "Impala", "Flink",
    "Storm", "Beam", "Samza", "Airflow", "Dagster", "Prefect", "Luigi", "dbt", "Fivetran",
    "Airbyte", "Stitch", "Kinesis", "Firehose", "Glue", "EMR", "Pinecone", "Weaviate",
    "Milvus", "Qdrant", "Chroma", "Faiss", "LanceDB", "PGVector", "Zilliz",
)

# ---- cloud: AWS ----
_add(
    "AWS", "EC2", "S3", "Lambda", "RDS", "Aurora", "ElastiCache", "DynamoDB", "Redshift",
    "EBS", "EFS", "FSx", "Glacier", "VPC", "Route53", "CloudFront", "CloudFormation",
    "CloudWatch", "CloudTrail", "IAM", "KMS", "SQS", "SNS", "SES", "EventBridge",
    "StepFunctions", "Fargate", "ECS", "EKS", "ECR", "Batch", "Lightsail", "Beanstalk",
    "Amplify", "AppSync", "Cognito", "Athena", "Glue", "EMR", "Kinesis", "Firehose",
    "QuickSight", "SageMaker", "Bedrock", "Comprehend", "Rekognition", "Polly", "Lex",
    "Transcribe", "Translate", "Textract", "Kendra", "Forecast", "Personalize", "Neptune",
    "DocumentDB", "Timestream", "Keyspaces", "MSK", "MQ", "AppFlow", "DataSync", "Snowball",
    "StorageGateway", "DirectConnect", "GlobalAccelerator", "AppMesh", "XRay", "Config",
    "SecretsManager", "SystemsManager", "WAF", "Shield", "GuardDuty", "Macie", "Inspector",
    "SecurityHub", "Organizations", "ControlTower", "Outposts", "Wavelength", "LocalZones",
)

# ---- cloud: GCP / Azure / others ----
_add(
    "GCP", "GKE", "ComputeEngine", "AppEngine", "CloudRun", "CloudFunctions", "CloudStorage",
    "BigQuery", "Bigtable", "CloudSQL", "Spanner", "Datastore", "PubSub", "Dataflow",
    "Dataproc", "Composer", "VertexAI", "CloudBuild", "ArtifactRegistry", "CloudCDN",
    "Azure", "AzureAD", "AKS", "AzureFunctions", "BlobStorage", "CosmosDB", "AzureSQL",
    "AzureDevOps", "AppService", "ServiceBus", "EventHub", "Synapse", "Databricks",
    "AzureML", "LogicApps", "APIM", "Bicep", "ARM",
    "Cloudflare", "Workers", "R2", "DigitalOcean", "Droplet", "Linode", "Vultr", "Hetzner",
    "Heroku", "Vercel", "Netlify", "Railway", "Render", "Fly.io", "OVH", "Scaleway",
)

# ---- containers / devops / CI-CD / IaC / observability ----
_add(
    "Docker", "Dockerfile", "DockerCompose", "Kubernetes", "k8s", "kubectl", "Kubeadm",
    "Helm", "Kustomize", "Istio", "Linkerd", "Envoy", "Consul", "Nomad", "Vault", "Terraform",
    "Terragrunt", "Pulumi", "Packer", "Vagrant", "Ansible", "Chef", "Puppet", "SaltStack",
    "Podman", "containerd", "CRIO", "runc", "Buildah", "Skaffold", "Tilt", "Minikube",
    "Kind", "k3s", "Rancher", "OpenShift", "Nginx", "Apache", "HAProxy", "Traefik", "Caddy",
    "Jenkins", "CircleCI", "TravisCI", "GitHubActions", "GitLabCI", "ArgoCD", "Flux", "Spinnaker",
    "Drone", "TeamCity", "Bamboo", "Concourse", "Tekton", "Harness", "Buildkite",
    "Git", "GitHub", "GitLab", "Bitbucket", "Gitea", "Gerrit", "SVN", "Mercurial",
    "Prometheus", "Grafana", "Loki", "Tempo", "Jaeger", "Zipkin", "Datadog", "NewRelic",
    "Dynatrace", "AppDynamics", "Splunk", "Kibana", "Logstash", "Fluentd", "Fluentbit",
    "Sentry", "Rollbar", "Bugsnag", "PagerDuty", "Opsgenie", "VictorOps", "Statsd",
    "OpenTelemetry", "Honeycomb", "Lightstep", "SonarQube", "Snyk", "Trivy", "Falco",
)

# ---- protocols / security / networking / web ----
_add(
    "API", "REST", "RESTful", "GraphQL", "gRPC", "tRPC", "SOAP", "WebSocket", "Webhook",
    "HTTP", "HTTPS", "HTTP2", "HTTP3", "QUIC", "TCP", "UDP", "IP", "IPv4", "IPv6", "ICMP",
    "DNS", "DHCP", "NAT", "CIDR", "BGP", "MQTT", "AMQP", "SMTP", "IMAP", "POP3", "FTP",
    "SFTP", "SSH", "Telnet", "VPN", "WireGuard", "OpenVPN", "TLS", "SSL", "mTLS", "CORS",
    "CSRF", "XSS", "SQLi", "DDoS", "WAF", "JWT", "OAuth", "OAuth2", "OIDC", "SAML", "LDAP",
    "Kerberos", "RBAC", "ABAC", "MFA", "2FA", "SSO", "PKI", "RSA", "AES", "SHA", "MD5",
    "HMAC", "PGP", "GPG", "CDN", "CDNs", "LoadBalancer", "ReverseProxy", "Proxy", "Gateway",
    "JSON", "XML", "YAML", "TOML", "CSV", "TSV", "Protobuf", "Avro", "Parquet", "ORC",
    "HTML", "HTML5", "CSS", "CSS3", "SVG", "WebGL", "WebRTC", "WebAssembly", "PWA", "SPA",
    "SSR", "SSG", "ISR", "CSR", "DOM", "BOM", "AJAX", "SSE", "CRUD", "MVC", "MVVM", "ORM",
)

# ---- engineering concepts / metrics / roles ----
_add(
    "p50", "p90", "p95", "p99", "p999", "latency", "throughput", "bandwidth", "uptime",
    "downtime", "QPS", "RPS", "TPS", "IOPS", "TTL", "RTT", "SLA", "SLO", "SLI", "MTTR",
    "MTBF", "KPI", "OKR", "ROI", "DAU", "MAU", "CTR", "CAC", "LTV", "ARR", "MRR",
    "PRD", "RFC", "ADR", "MVP", "POC", "Spec", "Backlog", "Sprint", "Standup", "Retro",
    "Kanban", "Scrum", "Agile", "Waterfall", "Epic", "Story", "Ticket", "Bug", "Hotfix",
    "Patch", "Release", "Rollout", "Rollback", "Canary", "BlueGreen", "FeatureFlag",
    "Cache", "Caching", "CDN", "Sharding", "Replication", "Partitioning", "Indexing",
    "Throttling", "RateLimit", "Backpressure", "Idempotent", "Idempotency", "Concurrency",
    "Parallelism", "Mutex", "Semaphore", "Deadlock", "RaceCondition", "Atomic", "Lock",
    "Queue", "Stack", "Heap", "Hashmap", "Trie", "Graph", "Tree", "BinarySearch", "BigO",
    "Microservice", "Microservices", "Monolith", "Serverless", "EventDriven", "PubSub",
    "Backend", "Frontend", "Fullstack", "DevOps", "MLOps", "DataOps", "SecOps", "SRE",
    "QA", "QC", "UAT", "Smoke", "Regression", "E2E", "Unit", "Integration", "Mock", "Stub",
    "SDK", "CLI", "GUI", "TUI", "IDE", "API", "ABI", "AST", "IR", "JIT", "AOT", "GC",
    "OS", "Kernel", "Syscall", "Daemon", "Cron", "Cronjob", "Thread", "Process", "Coroutine",
    "Async", "Await", "Promise", "Callback", "Closure", "Recursion", "Refactor", "Linting",
    "Compile", "Transpile", "Bundle", "Minify", "Lint", "Debugger", "Breakpoint", "Stacktrace",
    "Endpoint", "Middleware", "Namespace", "Schema", "Migration", "Seed", "Fixture", "DTO",
)

# ---- linux / shell / cli tooling ----
_add(
    "grep", "egrep", "ripgrep", "awk", "sed", "cut", "sort", "uniq", "xargs", "find", "tar",
    "gzip", "curl", "wget", "scp", "rsync", "ssh", "sshd", "chmod", "chown", "sudo", "systemctl",
    "journalctl", "crontab", "htop", "tmux", "screen", "vim", "nano", "emacs", "ps", "kill",
    "nohup", "lsof", "netstat", "ss", "ping", "traceroute", "dig", "nslookup", "iptables",
    "ufw", "make", "gcc", "clang", "gdb", "valgrind", "strace", "ltrace", "perf", "nvidia-smi",
    "npm", "npx", "yarn", "pnpm", "bun", "pip", "pipx", "uv", "cargo", "rustc", "go", "gofmt",
    "javac", "mvn", "gradle", "dotnet", "composer", "bundler", "gem", "brew", "apt", "yum",
    "dnf", "pacman", "snap", "flatpak", "kubectl", "helm", "docker", "git", "gh", "aws", "gcloud",
    "az", "terraform", "ansible", "vagrant", "jq", "yq", "fzf", "bat", "exa", "fd", "tldr",
)

# ---- file formats / extensions ----
_add(
    "PDF", "DOCX", "XLSX", "PPTX", "CSV", "TSV", "JSON", "JSONL", "XML", "YAML", "YML",
    "TOML", "INI", "ENV", "MD", "RST", "TXT", "LOG", "SQL", "SH", "PY", "JS", "TS", "TSX",
    "JSX", "GO", "RS", "RB", "PHP", "JAVA", "CPP", "PNG", "JPG", "JPEG", "GIF", "SVG", "WEBP",
    "MP3", "MP4", "WAV", "FLAC", "OGG", "AVI", "MKV", "ZIP", "TAR", "GZ", "RAR", "ISO",
    "DEB", "RPM", "APK", "DMG", "EXE", "DLL", "SO", "JAR", "WAR", "WASM", "ONNX", "GGUF",
)

# ---- products / SaaS / tools ----
_add(
    "Cursor", "Codex", "Copilot", "Tabnine", "Codeium", "Windsurf", "Cline", "Aider",
    "VSCode", "IntelliJ", "PyCharm", "WebStorm", "GoLand", "RubyMine", "CLion", "Xcode",
    "AndroidStudio", "Eclipse", "NetBeans", "Sublime", "Atom", "Notepad++", "Zed",
    "Jira", "Confluence", "Trello", "Asana", "Linear", "Notion", "ClickUp", "Monday",
    "Slack", "Teams", "Discord", "Zoom", "Figma", "Sketch", "Miro", "Loom", "Canva",
    "Postman", "Insomnia", "Swagger", "OpenAPI", "Stoplight", "Hoppscotch",
    "Stripe", "Razorpay", "PayPal", "Twilio", "SendGrid", "Mailgun", "Auth0", "Okta",
    "Clerk", "Supabase", "PlanetScale", "Neon", "Upstash", "Segment", "Amplitude",
    "Mixpanel", "Hotjar", "Posthog", "LaunchDarkly", "Optimizely", "Salesforce", "HubSpot",
    "Zendesk", "Intercom", "Freshdesk", "Tableau", "Looker", "PowerBI", "Metabase", "Superset",
)

# ---- programmatic version expansion (extra real variants) ----
for _b, _vers in {
    "GPT": ("Turbo", "Vision", "Mini", "Nano"),
    "Claude": ("Instant", "V2", "V3", "V4"),
    "Llama": ("Guard", "Instruct", "Chat"),
    "Gemini": ("Nano", "Flash8B", "Exp"),
    "Qwen": ("Max", "Plus", "Turbo", "Coder"),
    "Mistral": ("Large", "Medium", "Small", "Nemo"),
}.items():
    for _v in _vers:
        _add(_b + _v)

# =============================================================================
# 2. PHRASE_HINTS  (>= 500 Hinglish tech phrases, generated from real patterns)
# =============================================================================
_EN_ACTIONS = [
    "deploy", "rollback", "build", "merge", "commit", "push", "pull", "clone", "fork",
    "debug", "test", "review", "refactor", "restart", "reboot", "scale", "ship", "release",
    "revert", "rebase", "checkout", "configure", "install", "uninstall", "update", "upgrade",
    "downgrade", "rerun", "retry", "cache", "index", "query", "migrate", "monitor", "log",
    "trace", "profile", "benchmark", "optimize", "validate", "lint", "format", "compile",
    "run", "kill", "restart", "spawn", "scale", "patch", "fix", "review", "approve", "deploy",
]
_HINDI_FRAMES = [
    "{a} karna", "{a} karo", "{a} kar do", "{a} kar diya", "{a} mat karo", "{a} kar lo",
    "{a} karna hai", "{a} ho gaya", "{a} ho raha hai", "{a} ho jayega", "{a} ho gaya hai",
    "{a} fail ho gaya", "{a} fail ho raha hai", "{a} pending hai", "{a} complete ho gaya",
    "{a} kaise karu", "{a} kar diya hai", "{a} nahi hua", "{a} hone do", "{a} karke dekho",
]
_NOUN_PHRASES = [
    "docker image", "docker container build karo", "docker image push karo", "kubernetes pe deploy",
    "aws pe deploy karna hai", "aws pe", "aws se", "aws pe chala do", "ec2 pe deploy",
    "lambda function", "s3 bucket me daal do", "github pe push karo", "github pe", "gitlab pe",
    "cursor se", "cursor me likho", "cursor se code karo", "repo clone karo", "repo pull karo",
    "branch banao", "branch checkout karo", "merge conflict aa gaya", "pull request banao",
    "server restart karo", "server down hai", "server pe deploy", "api call kar raha hai",
    "api fail ho raha hai", "redis cache clear karo", "kafka topic", "postgres query slow hai",
    "mongodb me daal do", "build pipeline fail", "build fail ho gaya", "build pass ho gaya",
    "issue aa raha hai", "bug aa gaya", "error aa raha hai", "log check karo", "logs dekho",
    "deployment ho gaya", "rollback mat karo", "production pe mat karo", "staging pe test karo",
    "latency badh gayi", "p95 high hai", "p99 spike aa gaya", "throughput kam hai",
    "memory leak ho raha hai", "cpu spike aa gaya", "downtime ho gaya", "alert aa gaya",
    "code review pending hai", "prd update karo", "ticket close karo", "sprint plan karo",
    "model train ho raha hai", "gpu pe chala do", "inference slow hai", "prompt change karo",
    "token limit aa gaya", "embedding bana do", "rag setup karo", "fine tune karna hai",
]


def _build_phrase_hints() -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for frame in _HINDI_FRAMES:
        for a in _EN_ACTIONS:
            p = frame.format(a=a)
            if p not in seen:
                seen.add(p)
                out.append(p)
    for p in _NOUN_PHRASES:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


PHRASE_HINTS: tuple[str, ...] = _build_phrase_hints()


# =============================================================================
# 3. get_initial_prompt()  — Whisper vocabulary-biasing prompt
# =============================================================================
# A curated, high-value subset (Whisper's initial_prompt has a limited token budget,
# so we bias with the most impactful terms rather than the full 1000-entry dictionary).
_PROMPT_TERMS = [
    "GPT", "GPT4", "GPT5", "Claude", "Gemini", "Qwen", "Llama", "Mistral", "DeepSeek",
    "RAG", "LoRA", "QLoRA", "embedding", "transformer", "tokenizer", "PyTorch", "CUDA",
    "Docker", "Kubernetes", "Redis", "Kafka", "Postgres", "MongoDB", "MySQL", "GitHub",
    "GitLab", "Cursor", "Codex", "Copilot", "PRD", "API", "GraphQL", "backend", "frontend",
    "microservice", "webhook", "endpoint", "AWS", "EC2", "Lambda", "S3", "Azure", "GCP",
    "p50", "p95", "p99", "latency", "throughput", "rollback", "deploy", "staging", "production",
]
_PROMPT_PHRASES = [
    "deploy karna", "rollback mat karo", "build fail ho gaya", "docker image",
    "cursor se", "aws pe", "repo clone karo", "issue aa raha hai",
]


def get_initial_prompt() -> str:
    """Return a vocabulary-biasing prompt string for Whisper's `initial_prompt`."""
    terms = ", ".join(_PROMPT_TERMS)
    phrases = "; ".join(f"'{p}'" for p in _PROMPT_PHRASES)
    return (
        "This audio is technical work dictation that mixes English and Hindi (Hinglish). "
        f"It may contain terms such as {terms}. "
        f"It may also contain Hinglish phrases like {phrases}."
    )


# =============================================================================
# 4. normalize_tech_words()
# =============================================================================
_TECH_LOOKUP = {k: v for k, v in TECH_TERMS.items()}

# multiword ASR splits → canonical
_MULTIWORD = {
    "deep seek": "DeepSeek", "deep mind": "DeepMind", "hugging face": "HuggingFace",
    "git hub": "GitHub", "get hub": "GitHub", "git lab": "GitLab", "bit bucket": "Bitbucket",
    "open ai": "OpenAI", "my sql": "MySQL", "my sequel": "MySQL", "no sql": "NoSQL",
    "mongo db": "MongoDB", "post gres": "Postgres", "postgres ql": "PostgreSQL",
    "elastic search": "Elasticsearch", "type script": "TypeScript", "java script": "JavaScript",
    "node js": "Node.js", "next js": "Next.js", "nest js": "NestJS", "vue js": "Vue",
    "graph ql": "GraphQL", "rest api": "REST API", "web socket": "WebSocket",
    "web hook": "Webhook", "front end": "frontend", "back end": "backend",
    "micro service": "microservice", "micro services": "microservices", "data base": "database",
    "lang chain": "LangChain", "llama index": "LlamaIndex", "py torch": "PyTorch",
    "tensor flow": "TensorFlow", "scikit learn": "scikit-learn", "data dog": "Datadog",
    "pager duty": "PagerDuty", "cloud front": "CloudFront", "dynamo db": "DynamoDB",
    "cloud watch": "CloudWatch", "load balancer": "load-balancer", "vs code": "VSCode",
    "q lora": "QLoRA", "chat gpt": "ChatGPT", "gpt four": "GPT4", "gpt five": "GPT5",
    "claude three": "Claude3", "llama three": "Llama3", "ec two": "EC2", "s three": "S3",
}
_MULTIWORD_RE = [
    (re.compile(r"\b" + re.escape(k).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE), v)
    for k, v in sorted(_MULTIWORD.items(), key=lambda kv: -len(kv[0]))
]

_P_METRIC_RE = re.compile(r"(?i)\bp\s*-?\s*(50|90|95|99|999)\b")

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_MODEL_BASES = {"gpt", "claude", "llama", "gemini", "qwen", "mistral", "mixtral", "deepseek"}

# Curated unambiguous terms safe to recase in free text (avoids clobbering English words
# like go/rust/swift/node/spring/react/rest/cache/token/model/prompt/build/release...).
_RECASE_SAFE = (
    "AWS", "EC2", "S3", "GCP", "GPT", "GPT4", "GPT5", "Claude", "Gemini", "Qwen", "Llama",
    "Mistral", "DeepSeek", "QLoRA", "LoRA", "RAG", "Docker", "Kubernetes", "Redis", "Kafka",
    "Postgres", "PostgreSQL", "MongoDB", "MySQL", "GitHub", "GitLab", "Cursor", "Codex",
    "Copilot", "PRD", "API", "GraphQL", "PyTorch", "TensorFlow", "CUDA", "Kubectl", "Helm",
    "Nginx", "Terraform", "Ansible", "Jenkins", "Grafana", "Prometheus", "Datadog", "Jira",
    "Elasticsearch", "ClickHouse", "Snowflake", "Databricks", "HuggingFace", "OpenAI",
    "Anthropic", "Lambda", "DynamoDB", "CloudFront", "SageMaker", "Bedrock", "Whisper",
    "Parakeet", "Embedding", "Transformer", "Tokenizer", "JWT", "OAuth", "p50", "p90",
    "p95", "p99", "Webhook", "Microservice", "WebSocket", "DevOps",
)
_RECASE_SAFE_LOWER = {w.lower(): w for w in _RECASE_SAFE}

_PUNCT = ".,!?;:\"'()[]{}…—–"


def _split_trail(tok: str) -> tuple[str, str]:
    core = tok.rstrip(_PUNCT)
    trail = tok[len(core):]
    return core, trail


def _as_digit(tok: str):
    low = tok.lower()
    if low in _NUMBER_WORDS:
        return _NUMBER_WORDS[low]
    if tok.isdigit():
        return tok
    if re.fullmatch(r"\d+\.\d+", tok):
        return tok
    return None


def normalize_tech_words(text: str) -> str:
    """Normalize spaced / spelled-out tech tokens to canonical forms.

    'g p t four'->'GPT4', 'p 95'->'p95', 'a w s'->'AWS', 'q lora'->'QLoRA',
    'deep seek'->'DeepSeek', plus recasing of unambiguous known terms.
    """
    if not text:
        return text
    # 1) multiword phrases
    for pat, repl in _MULTIWORD_RE:
        text = pat.sub(repl, text)
    # 2) p-metrics
    text = _P_METRIC_RE.sub(lambda m: "p" + m.group(1), text)

    # 3) token pass: model+version joins, single-letter-run acronyms, single-letter+word, recase
    toks = text.split()
    out: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        core, trail = _split_trail(toks[i])
        low = core.lower()

        # model base + version  ("gpt four" -> GPT4)
        if low in _MODEL_BASES and i + 1 < n:
            nxt_core, nxt_trail = _split_trail(toks[i + 1])
            d = _as_digit(nxt_core)
            if d is not None:
                cand = low + d
                canon = _TECH_LOOKUP.get(cand) or (_TECH_LOOKUP.get(low, core) + d)
                out.append(canon + nxt_trail)
                i += 2
                continue

        # single-letter run  ("a w s" -> AWS, "g p t four" -> GPT4)
        if len(low) == 1 and low.isalpha():
            j = i
            letters: list[str] = []
            while j < n:
                c, _ = _split_trail(toks[j])
                if len(c) == 1 and c.isalpha():
                    letters.append(c.lower())
                    j += 1
                else:
                    break
            if len(letters) >= 2:
                base = "".join(letters)
                ver, extra = "", 0
                if j < n:
                    c, _ = _split_trail(toks[j])
                    d = _as_digit(c)
                    if d is not None:
                        ver, extra = d, 1
                cand = base + ver
                if cand in _TECH_LOOKUP:
                    last_trail = _split_trail(toks[j - 1 + extra])[1]
                    out.append(_TECH_LOOKUP[cand] + last_trail)
                    i = j + extra
                    continue
                if base in _TECH_LOOKUP:
                    last_trail = _split_trail(toks[j - 1 + extra])[1]
                    out.append(_TECH_LOOKUP[base] + ver + last_trail)
                    i = j + extra
                    continue
                # not a known acronym — emit letters unchanged
                out.append(toks[i])
                i += 1
                continue
            # single letter + following word  ("q lora" -> QLoRA)
            if len(letters) == 1 and j < n:
                nc, nt = _split_trail(toks[j])
                cand = low + nc.lower()
                if cand in _TECH_LOOKUP:
                    out.append(_TECH_LOOKUP[cand] + nt)
                    i = j + 1
                    continue

        # recase known unambiguous term
        if low in _RECASE_SAFE_LOWER:
            out.append(_RECASE_SAFE_LOWER[low] + trail)
            i += 1
            continue

        out.append(toks[i])
        i += 1
    return " ".join(out)


# =============================================================================
# 5. repair_common_asr_errors()
# =============================================================================
# (pattern, replacement, ignorecase) — ordered; conservative to avoid harming English.
_ASR_REPAIRS = [
    (r"\bcursor\s+(?:sir|sar|sur|sr)\b", "Cursor", True),
    (r"\bdockeri[sz]e\b", "Docker", True),
    (r"\bdockeri[sz]ed\b", "Docker", True),
    (r"\baws\s+p\b", "AWS pe", True),
    (r"\baws\s+pay\b", "AWS pe", True),
    (r"\bgit\s+hub\b", "GitHub", True),
    (r"\bget\s+hub\b", "GitHub", True),
    (r"\bgit\s+lab\b", "GitLab", True),
    (r"\bchat\s*gpt\b", "ChatGPT", True),
    (r"\bdeep\s+seek\b", "DeepSeek", True),
    (r"\bq\s+lora\b", "QLoRA", True),
    (r"\bquola\b", "QLoRA", True),
    (r"\bmy\s+sequel\b", "MySQL", True),
    (r"\bmy\s+sql\b", "MySQL", True),
    (r"\bpost\s+gres\b", "Postgres", True),
    (r"\bred\s+is\b", "Redis", True),
    (r"\bmongo\s+db\b", "MongoDB", True),
    (r"\bkuber\s*netes\b", "Kubernetes", True),
    (r"\bk\s+eight\s+s\b", "k8s", True),
    (r"\bk8s\b", "k8s", True),
    (r"\bnode\s+js\b", "Node.js", True),
    (r"\bnext\s+js\b", "Next.js", True),
    (r"\bgraph\s+ql\b", "GraphQL", True),
    (r"\bjason\b", "JSON", True),
    (r"\byou\s*are\s*el\b", "URL", True),
    (r"\bec\s+two\b", "EC2", True),
    (r"\bs\s+three\b", "S3", True),
    (r"\bprd\b", "PRD", True),
    (r"\bapi\b", "API", True),
    (r"\baws\b", "AWS", True),
    (r"\bgcp\b", "GCP", True),
    (r"\bgpt\b", "GPT", True),
    (r"\bllm\b", "LLM", True),
    (r"\brag\b", "RAG", False),     # case-sensitive: don't recapitalize the English word "rag"
    (r"\blora\b", "LoRA", False),
    (r"\bp\s*95\b", "p95", True),
    (r"\bp\s*99\b", "p99", True),
]
_ASR_REPAIRS_COMPILED = [
    (re.compile(p, re.IGNORECASE if ic else 0), r) for p, r, ic in _ASR_REPAIRS
]


def repair_common_asr_errors(text: str) -> str:
    """Fix frequent ASR mishearings of tech vocabulary.

    'cursor sir'->'Cursor', 'prd update'->'PRD update', 'aws p'->'AWS pe', 'dockerize'->'Docker'.
    """
    if not text:
        return text
    for pat, repl in _ASR_REPAIRS_COMPILED:
        text = pat.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    print("TECH_TERMS:", len(TECH_TERMS))
    print("PHRASE_HINTS:", len(PHRASE_HINTS))
    print("\ninitial_prompt:\n ", get_initial_prompt())

    print("\n-- normalize_tech_words --")
    for s in ["g p t four", "p 95", "a w s", "q lora", "deep seek",
              "use g p t four with q lora", "deploy on a w s with p 95 latency"]:
        print(f"  {s!r:45s} -> {normalize_tech_words(s)!r}")

    print("\n-- repair_common_asr_errors --")
    for s in ["cursor sir", "prd update", "aws p", "dockerize",
              "open the prd and deploy on aws p"]:
        print(f"  {s!r:40s} -> {repair_common_asr_errors(s)!r}")

    print("\nsample phrases:", PHRASE_HINTS[:6])
