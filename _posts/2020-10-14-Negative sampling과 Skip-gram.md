# Negative Sampling & Skip-gram

## Negative Sampling

* 배경

  * Skip-gram(Mikolov et at., 2013a)
    * target word가 주어졌을 때, context word가 무엇일 지 맞추는 과정에서 학습 	
    * 출력층에서 보통 수십만개가 되는 어휘집합의 단어의 softamx 계산량이 비교적 큼
    * 정답 context word의 나타날 확률은 높이고 나머지는 낮춰야 함

* Negative Sampling을 적용한 skip-gram(Mikolov et at., 2013b)

  * target word와 context word 쌍이 주어졌을 때, 해당 쌍이 positive sample인지 negative sample인지 binary classfication하는 과정에서 학습

    * (positive) sample: target word(t)와 실제로 주변에 등장한 context vector(c)의 쌍

    * negative sample:target word(t)와 그 주변에 등장하지 않은 단어(말뭉치 전체에서 랜덤 추출)의 쌍

    * 예시

      | 에서  | 이불  | 빨래 | 를    | 하는  | 가족 |
      | ----- | ----- | ---- | ----- | ----- | ---- |
      | $c_1$ | $c_2$ | t    | $c_3$ | $c_4$ | -    |

      * positive sample

        | t    | c    |
        | ---- | ---- |
        | 빨래 | 에서 |
        | 빨래 | 이불 |
        | 빨래 | 를   |
        | 빨래 | 하는 |

      * negative sample

        |      | c    |
        | ---- | ---- |
        | 빨래 | 책상 |
        | 빨래 | 안녕 |
        | 빨래 | ...  |
        | 빨래 | 숫자 |

  * 기존 방법보다 계산량이 훨씬 적음

| 기존 계산량                                                  | Negative Sampling 계산량                           |
| ------------------------------------------------------------ | -------------------------------------------------- |
| 모델을 1step에 전체 단어를 모두 계산                         | 1개 positive sample과 k개의 negative sample만 계산 |
| =매 step마다 어휘 집합 크기 만큼 차원수를 가진 softmax를 1회 계산 | =매 step마다 차원수가 2인 sigmoid를 k+1회 계산     |

* 작은 데이터에서는 k=5~20, 큰 말뭉치에서는 k=2~5로 하는 것이 성능이 좋음

* nagetive sampling 방법

  * **말뭉치에 자주 등장하지 않은 희귀한 단어가 조금 더 잘 뽑힐 수 있도록 설계**

  * negative sample 확률

    $P_{negative}(w_i) = \frac{U(w_i)^{3/4}} {\sum^n_{j=0}{U(w_j)^{3/4}}} $

    $U(w_j)$: 해당 단어의 unigram 확률(해당 단어 빈도/전체 단어 수)

  * 예시
    * 말뭉치 안에 단어가 사과, 바나나 둘뿐이고 비율은 0.99, 0.01일 경우
    * 각각 nagative로 뽑힐 확률
      * $P(사과) = \frac{0.99^{0.75}}{0.99^{0.75}+0.01^{0.75}} = 0.97$ //낮아짐
      * $P(바나나) = \frac{0.01^{0.75}}{0.99^{0.75}+0.01^{0.75}} = 0.03$ //높아짐
      * --> 분자만 보자, 각 확률의 0.75씩 제곱해줘서 큰거는 작게, 아주 작은값은 키워주나? 



## Skip-gram

