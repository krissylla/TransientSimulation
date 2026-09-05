from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

from pathlib import Path

project_dir = Path(__file__).resolve().parent
input_dir = project_dir / "results_5"
output_file = input_dir / "all_sources.parquet"


# ------------------------------------------------------------
# Find input parquet files
# ------------------------------------------------------------

files = sorted(input_dir.glob("detected_N*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No input parquet files found in {input_dir}"
    )

# Don't accidentally include the output file if the script is rerun
files = [
    f for f in files
    if f.name != output_file.name
]

print(f"Found {len(files)} parquet files.")
print(f"Output: {output_file}")


# ------------------------------------------------------------
# Check that schemas are consistent
# ------------------------------------------------------------

first_file = pq.ParquetFile(files[0])
schema = first_file.schema_arrow

print("\nSchema:")
print(schema)


# ------------------------------------------------------------
# Combine files without loading everything into memory
# ------------------------------------------------------------

tmp_file = output_file.with_suffix(".tmp.parquet")

writer = pq.ParquetWriter(
    tmp_file,
    schema=schema,
    compression="zstd"
)

total_rows = 0

try:

    for i, file in enumerate(files, start=1):

        print(
            f"[{i}/{len(files)}] "
            f"{file.name}"
        )

        parquet_file = pq.ParquetFile(file)

        # Read one record batch at a time
        for batch in parquet_file.iter_batches(
            batch_size=100_000
        ):

            table = pa.Table.from_batches([batch])

            # Make sure schema is compatible
            if not table.schema.equals(schema):
                raise ValueError(
                    f"Schema mismatch in {file}"
                )

            writer.write_table(table)

            total_rows += batch.num_rows

finally:
    writer.close()


# ------------------------------------------------------------
# Rename temporary file only after successful completion
# ------------------------------------------------------------

tmp_file.replace(output_file)

print("\nDone!")
print(f"Total rows: {total_rows:,}")
print(f"Output: {output_file}")