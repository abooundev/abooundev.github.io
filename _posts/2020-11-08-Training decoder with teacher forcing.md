---
title: "Training decoder with teacher forcing"
date: 2020-10-11
math: true
---



# Training decoder with teacher forcing 

Seq2seq는 RNN으로 이루어진 encoder와 decoder의 구조를 갖습니다. encoder는 입력된 input sentence를 context로 encoding하고, decoder는 전달 받은 context를 가지고 decoding하여 output sequence을 출력하게 됩니다. seq2seq 구조는, machine translation 같은 문제에 응용할 수 있습니다. 

![image](https://user-images.githubusercontent.com/65707664/98466466-f596f980-2212-11eb-949e-79e14a913d9d.png)

* 그림 출처: cs224n 

### Teacher forcing이란?

decoder는 첫 step의 입력으로 \<BOS\>토큰을 주고, 출력된 output을 다시 다음 step의 입력으로 넣어주는 과정을 만복하며, 모델이 최종적으로 \<EOS\>를 출력하면 decoding 과정을 종료하게 됩니다. 

그러나 실제로 모델의 training 과정에서 \<BOS\>만을 가지고, decoder가 training data를 학습하는 것은 쉽지 않습니다. (정확히는 수렴 속도가 오래걸리게 됩니다.) 따라서 training 과정에서 teacher forcing을 적용할 수 있습니다.

Teacher forcing은 Tensorflow seq2seq tutorial에서 다음과 같이 설명하고 있습니다.

> **Teacher forcing** is the technique where the **target word** **is passed as the** **next input** **to the decoder**. 

	* 출처: https://www.tensorflow.org/tutorials/text/nmt_with_attention

Decoder는 training 과정에서 입력으로 이전 step의 예측값을 받는 것이 아닌, 타겟값을 받게 됩니다. decoder의 모델이 잘못된 값을 예측하더라도, 각 step에서 타겟값으로 넣어서 학습하는 것이죠. Teacher forcing을 적용해서, 빠른 수렴을 기대할 수 있을 것입니다.

<img width="661" alt="image" src="https://user-images.githubusercontent.com/65707664/98466472-05aed900-2213-11eb-8b29-c21c0e05fd44.png" style="zoom:150%;" >

* 그림 출처: https://github.com/tensorflow/nmt

Tensorflow seq2seq + attention tutorial을 통해 코드에 어떻게 적용되는지 볼 수 있습니다. 아래는 training 과정에서 teacher forcing을 적용한 코드 입니다. 

~~~python
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
~~~

Test에서는 teacher forcing을 적용하지 않습니다. 

~~~python
def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))
  sentence = preprocess_sentence(sentence)
  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''
  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    # not using teacher forcing here
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
~~~

* 코드 출처: https://www.tensorflow.org/tutorials/text/nmt_with_attention

### Teacher forcing을 적용하는 것이 항상 좋을까?

그러면 decoder를 training 시킬때, 항상 teacher forcing을 적용하는 것이 답일까요? Teacher forcing을 적용 decoding는 실제 문제들에 대해서도 잘 decoding 할까요? Ian Goodfellow 외 저술한 Deep Learning에서는 다음과 같이 한계에 대해 말하고 있습니다.

> The **disadvantage** of strict teacher forcing arises if the network is going to be later used in an closed loop mode, with the network outputs (or samples from the output distribution) fed back as input. In this case, the **fed-back inputs that the network sees during training** could be quite **diﬀerent from the kind of inputs that it will see at test time**. 

* 출처: https://www.deeplearningbook.org/contents/rnn.html

training 과정에서 본 input과 test에서 보게 될 input이 다를 경우, 실제 test에서의 성능은 training의 성능보다 떨어 질 수 있습니다. training data에 overfitting이 되버리면, 이 모델은 좋은 generalization을 가졌다고 보기 어려울 수 있습니다.  그러면 어떻게 빠르게 수렴하면서, 실제 문제에는 robust한 모델을 만들 수 있을까요?

### 어떻게 teacher forcing을 적용할까? 

> One way to mitigate this problem is to **train with both teacher-forced inputs and free-running inputs,** for example **by predicting the correct target a number of steps in the future through the unfolded recurrent output-to-input paths.** In this way, the network can learn to take into account input conditions (such as those it generates itself in the free-running mode) not seen during training and how to map the state back toward one that will make the network generate proper outputs after a few steps. 
>
> Another approach (Bengioet al., 2015b) to mitigate the gap between the inputs seen at training time and the inputs seen at test time **randomly chooses to use generated values or actual data values as input.** This approach exploits **a curriculum learning strategy** to gradually use more of the generated values as input.

출처: https://www.deeplearningbook.org/contents/rnn.html

한 가지 대안으로 모델에게 teacher forcing을 랜덤하게 적용할 수 있습니다. 랜덤값을 생성하여, 기준값 이상일 경우에는 예측값으로 훈련하고, 아닐 경우에는 타겟값으로 훈련하게 됩니다. 

아래는 pytorch의 seq2seq의 튜토리얼 입니다. 

~~~python
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
	 	decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
~~~

* 출처: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-the-model





[참고]

https://arxiv.org/pdf/1610.09038.pdf

