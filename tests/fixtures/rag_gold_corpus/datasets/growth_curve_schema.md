# column:OD750
OD750 is a numeric column in the growth-curve table.

# column:time_h
time_h is elapsed time measured in hours.

# column:sample_id
sample_id identifies the sample and is the grouping key for per-sample aggregation.

# column:temperature_c
temperature_c is required and is not nullable in the benchmark import schema.

# column:replicate_index
replicate_index identifies biological replicates.

# columns
The columns are sample_id, time_h, replicate_index, OD750, and temperature_c. generation_number is not a column.

# missing-values
Required numeric missing values are rejected during import rather than silently imputed.

# scope
The CSV is experimental evidence and cannot overwrite the current strain-state database.
