import os

import gpt_2_simple as gpt2

# Load model.
old_wd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sess = gpt2.start_tf_sess(threads=1)
gpt2.load_gpt2(sess)
os.chdir(old_wd)


def complete(prefix: str, length: int=25, include_prefix: bool=True) -> str:
    global sess
    old_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generated = gpt2.generate(
        sess,
        prefix=prefix,
        length=length,
        temperature=0.4,
        nsamples=1,
        return_as_list=True,
		include_prefix=include_prefix,
    )[0]
    os.chdir(old_wd)
    return generated


if __name__ == "__main__":
    prefixes = [
        "He's gonna go for the flip-reset",
        "And we're in the grand finals",
        "David passes to Robbie",
        "And we're at match point",
    ]

    for prefix in prefixes:
        # Generates the messages using the model.
        generated = gpt2.generate(
            sess,
            prefix=prefix,
            length=25,
            temperature=0.2,
            nsamples=5,
            return_as_list=True,
        )
        print("\n".join(generated))
        print()
