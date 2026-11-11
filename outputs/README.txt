LOCAL EXPERIMENT AND INSPECTION OUTPUTS -- excluded from Git by .gitignore

inspection/<run>/      Images, region JSON/CSV, standard inspection_report/ and inspection_report_ai/
evaluation/<run>/      Evaluation metrics, plots, standard paper_package/ and paper_package_ai/
audits/<run>/          Dataset audit records

01_MIGRATE_ALL.cmd copies historical outputs from earlier Studio versions, including
both standard and AI report folders, while preserving original data and collisions.
