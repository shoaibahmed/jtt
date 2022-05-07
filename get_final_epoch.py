import os
import sys
import natsort


if len(sys.argv) != 2:
    print("Usage: {sys.argv[0]} <Result directory>")
    exit()

res_dir = sys.argv[1]

assert os.path.exists(res_dir)
result_dir_contents = os.listdir(res_dir)
print("Result dir:", result_dir_contents)

assert "job.sh" in result_dir_contents
assert "model_outputs" in result_dir_contents

model_dir = os.path.join(res_dir, "model_outputs")
model_dir_contents = os.listdir(model_dir)
print("Model dir contents:", model_dir_contents[:5])

# Get all train CSV files
selected_files = natsort.natsorted([x for x in model_dir_contents if "output_train" in x])
print("Selected files:", len(selected_files), selected_files[-5:])

last_file = selected_files[-1]
print("Last file:", last_file)
final_epoch = int(last_file.replace("output_train_epoch_", "").replace(".csv", ""))
print("Final epoch:", final_epoch)
exit(final_epoch)
