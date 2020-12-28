import os

import gpt_2_simple as gpt2

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(
        model_name=model_name
    )  # Model is saved into current directory under /models/(model_name)/


dataset_name = "data"
file_name = dataset_name + ".txt"


# Comment this if you already have a pretrained model.
sess = gpt2.start_tf_sess()
gpt2.finetune(sess, file_name, model_name=model_name, steps=-1, sample_length=40)

# Uncomment this if using a model from already saved and not the one that is being finetuned above.
# sess = gpt2.start_tf_sess()
# gpt2.load_gpt2(sess, run_name=dataset_name)


# Generates the messages using the model.
generated = gpt2.generate(sess, length=40, temperature=0.2, nsamples=15, batch_size=5,)
