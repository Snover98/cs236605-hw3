r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # ====== YOUR CODE: ======
    start_seq = "ACT I"
    temperature = 0.75
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences so that we can use batches for several iterations instead of the whole epoch at once.  
This is for the same reason we use minibatches in normal deep learning problems -  
it has a similar accuracy and runs a lot faster.
"""

part1_q2 = r"""
It is possible because of the usage of the hidden state (which encodes context and position in the text)  
and because of the other parameters that the model learns.  
This means that it basically remembers the text and can generate it from certain spots depending on the context  
(i.e the hidden state).
"""

part1_q3 = r"""
We are not shuffling the batches because we do want to memorise the entire text in the correct order.  
This means that we want to use the batches in the same order every epoch and pass the hidden state from  
batch to batch within the same epoch.
"""

part1_q4 = r"""
**1.**  
We lower the temperature when sampling in relation to the 1.0 default for training because  
when training we want to see the actual confidence level that our model has for each output,  
while when we actually want to use it we only really want to consider the outputs that the model  
is really confident about - so we lower the temperature.  
  
**2.**  
When the temperature is very high we are basically using a uniform distribution,  
which happens because we make all of the logits close to 0 and the exponents for all of them is around 1.  
That means that for each output the softmax return about the same value, which is the same as a uniform distribution.  
  
**3.**  
When the temperature is very low we are basically not sampling but just taking the output  
with the highest confidence value deterministically.  
That happens because if a certain logit is higher than the rest, with high  temperature  
the difference between it and the others grows by a lot,  
meaning that after the exponent the other values should be so small in relation to it that  
after dividing by the sum (i.e normalizing to 1) they would have values around 0,  
while the highest logit would have a value around 1.  
This is basically the same as classifying with the model.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.2,
        discriminator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
When training, we are using a loss function for the discriminator and a loss function for the generator.  
When doing a backward pass on the discriminator loss the gradients try to improve it's ability to discern  
real inputs from ones the generator created.  
If we maintained the gradients for sampling in this case, we'd have gradients that would try and make  
the generator **worse** because that is what minimizing the discriminator loss function means.  
On the other hand, when we want to train the generator according to it's loss function,  
it's obvious that we'd want to maintain the gradients for it - after all we want to use them to improve the generator.  
That means that we only want to maintain gradients while sampling from the generator only  
when the samples would be used in the generator loss function in order to improve it.
"""

part3_q2 = r"""
**1.**  
When training a GAN to generate images we should **not** decide to stop only because the generator loss is below a threshold.  
That is because it's very much possible that both the discriminator and the generator improved this epoch in such a way  
that their losses stay the same, but the output is still of higher quality.  
In this case while the improvements in them cancel each other out in the loss function, it's still an improvement.  
  
**2.**  
If the discriminator loss stays at a constant value while the generator loss decrease it could mean that  
the rate of the discriminator improvement is lower then the rate of the generator improvement.  
Basically it means that the difference between the qualities of their outputs is getting larger.
"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


