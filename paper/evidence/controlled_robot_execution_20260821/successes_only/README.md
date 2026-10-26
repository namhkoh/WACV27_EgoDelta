# Success-Only Robot Execution Export

This directory contains only successful outcomes. combined_claimed_success_inventory.csv records 60 claimed successes: 30 harvesting and 30 deleafing. Of these, 40 have trial-level artifacts and 20 are historical operator-confirmed successes without recoverable trial IDs, logs, videos, or execution artifacts.

The 40 artifact-backed rows are classified by their frozen ledgers as direct execution with policy_dir=not_applicable, although the operator identifies the cohort as policy runs. This provenance conflict is preserved rather than silently relabeled. No artifact-backed learned-policy row was located.

This success-only view is descriptive and cannot establish a prospective denominator because rejected, failed, abandoned, and retried attempts are excluded. The complete attempt audit remains in the parent directory.
