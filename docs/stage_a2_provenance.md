# Stage A.2 provenance

Stage A.2 was executed and technically verified before this repository had its
initial Git commit. The original Stage A.2 artifacts therefore cannot truthfully
name a source commit, and they remain byte-for-byte frozen.

The initial commit
`16eea5a77b99ebe3c1831a65949859c58ec2251c` is a post-run snapshot of the
verified source tree used for Stage A.2. It is provenance evidence, not evidence
that the run was launched from an already committed checkout.

Post-Stage-A.2 diagnostics record this SHA and the hashes of the historical
receipts in a new diagnostic provenance receipt. They never edit the historical
Stage A, Stage A.1, or Stage A.2 receipts, checkpoints, configurations, or
candidate caches.

All post-run analyses are diagnostic only. They cannot promote the frozen QI
method, alter its scientific gate, or authorize access to the sealed test split.
