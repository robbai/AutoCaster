import os

import gpt_2_simple as gpt2

if __name__ == "__main__":
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(
            model_name=model_name
        )  # Model is saved into current directory under /models/(model_name)/

    dataset_name = "data"
    file_name = dataset_name + ".txt"

    # Train model.
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, file_name, model_name=model_name, steps=-1, sample_length=40)

    # Generates the messages using the model.
    generated = gpt2.generate(
        sess, length=40, temperature=0.2, nsamples=15, batch_size=5,
    )
