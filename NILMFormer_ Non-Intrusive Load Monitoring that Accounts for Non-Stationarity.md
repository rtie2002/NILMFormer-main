

arXiv:2506.05880v1  [cs.LG]  6 Jun 2025
NILMFormer: Non-Intrusive Load Monitoring that Accounts for
Non-Stationarity
## Adrien Petralia
EDF R&D - Université Paris Cité
## Paris, France
adrien.petralia@gmail.com
## Philippe Charpentier
## EDF R&D
## Palaiseau, France
philippe.charpentier@edf .f r
## Youssef Kadhi
## EDF R&D
## Palaiseau, France
youssef.kadhi@edf.fr
## Themis Palpanas
## Université Paris Cité
## Paris, France
themis@mi.parisdescartes.f r
## Abstract
Millions of smart meters have been deployed worldwide, collecting
the total power consumed by individual households. Based on these
data, electricity suppliers offer their clients energy monitoring so-
lutions to provide feedback on the consumption of their individual
appliances. Historically, such estimates have relied on statistical
methods that use coarse-grained total monthly consumption and
static customer data, such as appliance ownership. Non-Intrusive
Load Monitoring (NILM) is the problem of disaggregating a house-
hold’s collected total power consumption to retrieve the consumed
power for individual appliances. Current state-of-the-art (SotA)
solutions for NILM are based on deep-learning (DL) and operate on
subsequences of an entire household consumption reading. How-
ever, the non-stationary nature of real-world smart meter data
leads to a drift in the data distribution within each segmented win-
dow, which significantly affects model performance. This paper
introduces NILMFormer, a Transformer-based architecture that in-
corporates a new subsequence stationarization/de-stationarization
scheme to mitigate the distribution drift and that uses a novel posi-
tional encoding that relies only on the subsequence’s timestamp
information. Experiments with 4 real-world datasets show that
NILMFormer significantly outperforms the SotA approaches. Our
solution has been deployed as the backbone algorithm for EDF’s
(Electricité De France) consumption monitoring service, delivering
detailed insights to millions of customers about their individual
appliances’ power consumption. This paper appeared in KDD 2025.
CCS Concepts
•Hardware→Energy metering;•Computing methodologies
→Neural networks.
## Keywords
Non-Intrusive Load Monitoring, Deep-Learning, Non-Stationarity
ACM Reference Format:
Adrien Petralia, Philippe Charpentier, Youssef Kadhi, and Themis Palpanas.
- NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-
Stationarity. InProceedings of Make sure to enter the correct conference title
from your rights confirmation email (Conference acronym ’XX).ACM, New
York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
## ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX
## 0
## 5
## 4
## 3
## 2
## 1
## 0
## 5
## 4
## 3
## 2
## 1
kW
kW
## 푃퐷퐹푃퐷퐹
## Heater
## Water
## Heater
## Electric
## Vehicle
## Washer
## Household
smart meter
Aggregate household power consumption
Individual appliances power consumption
Figure 1: Illustration of a smart meter signal, composed of
the addition of each individual appliance running in the
household, and its non-stationary nature.
## 1  Introduction
Efficient energy management is emerging as a major lever for tack-
ling climate change. Thus, enabling consumers to understand and
control their own energy consumption is becoming crucial. In this
regard, millions of smart meters have been deployed worldwide in
individual households [8,41], recording the total household electric-
ity consumption at regular intervals (typically the average power
consumed every 10-30min, depending on the country [64]), pro-
viding valuable data that suppliers use to forecast energy demand.
Though, as Figure 1 shows, this recorded signal is the aggregated
power consumed byallappliances operating simultaneously.
EDF (Electricité De France), one of the world’s largest electricity
producers and suppliers, offers to clients a consumption monitoring
solution calledMon Suivi Conso, accessible via a web interface and
a mobile application [13]. A key feature of this service is providing
customers with details about individual appliance consumption.
Historically, only annual appliance-level feedback was available,
and estimates relied on semi-supervised statistical methods that
used the customers’ static information (e.g., appliance presence) and
total monthly consumption [14,15]. However, recent user feedback
has increasingly highlighted the wish to have access to fine-grained
information [11], prompting EDF to develop a more detailed, accu-
rate, and personalized solution for appliance consumption feedback
to enhance customer satisfaction and retention.
A recent internal study at EDF explored Time Series Extrinsic
Regression (TSER) approaches [56] to provide monthly appliance-
level feedback. Using these approaches, a model is trained to predict

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
the appliance’s total monthly consumption based on the monthly
smart meter signal, resulting in a marked improvement over pre-
vious semi-supervised methods, and reducing the mean absolute
error (MAE) of misassigned energy by an average of 70%. Despite
this progress, monthly-level feedback remains relatively coarse,
limiting its practical utility. Furthermore, recent studies suggest
that real-time awareness of energy-intensive appliance usage can
lower electricity consumption by up to 12% [2,4,12], highlighting
the importance of providing finer-grained details.
Non-Intrusive Load Monitoring (NILM) refers to the challenge of
estimating the power consumption, pattern, or on/off state activa-
tion of individual appliances using only the aggregate smart meter
household reading [29,35,52]. Since the introduction of Deep-
Learning (DL) for solving the NILM problem [30], many different
approaches have been proposed in the literature, ranging from
Convolutional-based [17, 18, 38, 54, 60, 63] to recent Transformer-
based architectures [1,55,57,61,66]. Nevertheless, to achieve suit-
able performances (in terms of cost and accuracy), all of the solu-
tions listed above shared the same setting: the DL algorithm takes
as input asubsequenceof an entire household reading to estimate
the individual appliances’ power. However, as shown in Figure 1,
the statistics (illustrated by the Probability Density Function) dras-
tically vary for two subsequences extracted from the same smart
meter reading. The intermittent run of electrical appliances causes
this phenomenon. For instance, in Figure 1, the heating system,
when active, consumes a substantial amount of energy, leading to
an increase in the overall power. This change in statistical prop-
erties over time is known as the data distribution shift problem
and is well known to hurt DL models’ performances for time series
analysis, especially in Time series forecasting (TSF) [33,36,43] In
TSF, the solutions operate in a setting similar to the NILM problem,
i.e., DL solutions are applied on subsequences of a long series. Re-
cent studies conducted in this area [33,36] have shown that taking
into account this data distribution drift is the key to achieving ac-
curate predictions. To the best of our knowledge, no studies have
investigated this issue in the context of energy disaggregation.
In this paper, we propose NILMFormer, a sequence-to-sequence
Transformer-based architecture designed to handle the data distri-
bution drift that occurs when operating on subsequences of an en-
tire household consumption reading. For this purpose, NILMFormer
employs two simple but effective mechanisms, drawing inspiration
from recent studies in TSF [33,36]. The first one consists of making
the input stationary by removing its mean and standard deviation)
and passing the removed statistics information as a new token to
the Transformer Block (referred to asTokenStats). The second one
involves learning a projection of the input subsequence’s statistics
(mean and std) to denormalize the output signal and predict the
individual appliance power (referred to asProjStats). Additionally,
NILMFormer employs TimeRPE, a Positional Encoding (PE) based
only on the subsequence’s timestamp information, which enables
the model to understand time-related appliance use. Overall, NILM-
Former is significantly more accurate than current State-of-the-Art
(SotA) NILM solutions, and drastically reduces the error of misas-
signed energy, compared to TSER methods (when used to provide
per-period feedback, i.e., daily, weekly, and monthly). NILMFormer
has been successfully deployed in EDF’s consumption monitoring
solutionMon Suivi Consoand currently provides millions of cus-
tomers with detailed insights about their individual appliances’
power consumption.
Our contributions can be summarized as follows:
•We propose NILMFormer, a sequence-to-sequence Transformer-
based architecture for energy disaggregation, especially designed
to handle the data distribution drift that occurs when operating on
subsequences of an entire consumption series. We also introduce
TimeRPE, a PE for Transformer-based model based only on the
subsequence’s timestamp discrete values, which significantly boosts
the performance of NILMFormer compared to standard PEs.
•We evaluate NILMFormer on 4 real-world datasets against SotA
NILM baselines and provide an in-depth analysis of the effective-
ness of the proposed solution to handle the aforementioned data
distribution drift problem.
## •
We provide insights about the deployment of our solution in EDF
Mon Suivi Consomonitoring solution, as well as highlighting the
value and practicality of the solution for EDF’s decision-making.
The source code of NILMFormer is available online [46].
2  Background and Related Work
2.1  Non-Intrusive Load Monitoring (NILM)
NILM [24], also called load disaggregation, relies on identifying
the individual power consumption, pattern, or on/off state acti-
vation of individual appliances using only the total aggregated
load curve [29,35,49,52]. Early NILM solutions involved Com-
binatorial Optimization (CO) to estimate the proportion of total
power consumption used by distinct active appliances at each time
step [24]. Later, unsupervised machine learning algorithms, such
as factorial hidden Markov Models (FHMM) were used [32]. NILM
gained popularity in the late 2010s, following the release of smart
meter datasets [21,31,34]. Kelly et al. [30] were the first to inves-
tigate DL approaches to tackle the NILM problem, and proposed
three different architectures, including a Bidirectional Long-Short-
Term Memory (BiLSTM). Zhang et al. [63] proposed a Fully Con-
volutional Network (FCN) and introduced the sequence-to-point
framework, which estimates the power consumption of individual
appliances only at the middle point of an input aggregate subse-
quence. However, this solution is not scalable to real-world long and
large datasets currently available to suppliers, as the model needs
to operate timestamp by timestamp over the entire consumption
series to predict the individual appliances’ power. Thus, numerous
sequence-to-sequence Convolutional-based architecture were later
investigated, e.g., Dilated Attention ResNet [60], Temporal Poool-
ing [38] and UNet [18]. With the breakthrough of the Transformer
architecture [58] in Natural Language Processing, BERT4NILM [61]
was proposed. Then, variants of these architectures was proposed,
such as Energformer [1] that introduced a convolutional layer in
the transformer block and replaced the original attention mecha-
nism with its linear attention variant for efficiency. More recently,
STNILM [57] is a variant of BERT4NILM that replaces the standard
FFN layer with the Mixture-Of-Expert layer [20]. In addition, hybrid
architectures were recently investigated, such as BiGRU [51] that
mixes Convolution layers and Recurrent Units, and TSILNet [66].
All the studies mentioned above apply their solution to the subse-
quences of an entire household reading. However, as pointed out in

NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-StationarityConference acronym ’XX, June 03–05, 2018, Woodstock, NY
Section 1, this leads to a large distribution drift in each subsequence,
hurting model performance. Despite the large number of studies
and different proposed architectures, none of them has addressed
this issue in the context of energy disaggregation. Moreover, and
mainly due to the lack of large publicly available datasets, none
of these studies provide insights into the performance of the pro-
posed baselines in real-world scenarios involving high-consuming
appliances, such as a Heating system, Water Heater, or Electric
## Vehicle.
2.2  Non-Stationarity in Time Series Analysis
DL is an established solution for tackling time series analysis tasks [22,
42,62], including electricity consumption series [37,47,48,50].
However, recent studies conducted in Time Series Forecasting (TSF)
pointed out that the non-stationary aspect of real-world time series
(e.g., the change in statistics over time) hurts model performance [33,
36,43]. Indeed, similar to the NILM setting, DL methods for TSF
operate on subsequences (called look-back windows); this leads
to data distribution drifts over the subsequences. To mitigate this
problem, RevIN [33], a stationarization/destationarization scheme,
is a solution adopted by most SotA TSF architectures [27,42]. RevIN
applies a per-subsequence normalization to the input, making each
series inside the network follow a similar distribution, and restores
the removed statistics in the network’s output. However, this pro-
cess can lead to a so-called over-stationarization, i.e., a loss of
information inside the network. The Non-Stationary Transformer
framework [36] mitigates this by combining RevIN with a hand-
crafted attention mechanism that feeds the removed subsequence
statistical information in the network. Despite their success in TSF,
applying such solutions directly to the NILM problem is not possi-
ble. Specifically, per-subsequence scaling like RevIN is not ideal for
NILM since power values are closely related to specific appliances.
Similar patterns (shapes) may be common to multiple appliances,
but the maximum power differentiates them. Moreover, restoring
input statistics in the output does not align with NILM objectives,
as disaggregation inherently involves a change in statistical values:
an appliance’s individual consumption is always lower than the
aggregate input signal. Thus, while RevIN and similar methods are
effective in TSF, they are not directly applicable to NILM.
[Data Drift Consideration in NILM]Mitigating the data drift
when solving the NILM problem has been studied in the past. In [23],
the authors propose a framework to mitigate data drift in high-
frequency measurements of electricity consumption. Their method
uses subsequences of real, apparent, and reactive power recorded
at much higher resolution than standard household smart meters,
which typically measure only average real power. Moreover, their
approach is based on classification based methods that are only
able to detect whether a subsequence (window of 6sec) corresponds
to an appliance’s usage (more akin to signature recognition) and
cannot be used to predict and estimate appliance-specific power
consumption. Meanwhile, Chang et al. [5] present a tree-based
method to adapt a trained model from one domain (eg, region or
household) to another—a transfer learning perspective. Thus, none
of these solutions tackle the intrinsic distribution drift observed
in subsequences of standard smart meter data during the training
phase.
2.3  Time Series Extrinsic Regression (TSER)
TSER solves a regression problem by learning the relationship
between  a  time  series  and  a  continuous  scalar  variable,  a  task
closely related to Time Series classification (TSC) [56]. A recent
study  [56]  compared  the  current  SotA  approaches  for  TSC  ap-
plied to TSER, and showed that CNN-based neural networks (Con-
vNet [44], ResNet [25], InceptionTime [19]), random convolutional
based (Rocket) [9] and XGBoost [6] are the current best approaches
for this task.
Note that predicting the total individual appliance power con-
sumed by an appliance over time (e.g., per day, week, or month) is
a straightforward application case of TSER.
## 3  Problem Definition
A smart meter signal is a univariate time series푥=(풙
## 1
## , ...,풙
## 푇
## )of
푇timestamped power consumption readings. The meter reading is
defined as the time differenceΔ
## 푡
## =푡
## 푖
## −푡
## 푖−1
between two consecu-
tive timestamps푡
## 푖
. Each element풙
## 푡
, typically measured in Watts
or Watt-hours, represents either the actual power at time푡or the
average power over the intervalΔ
## 푡
## .
[Energy Disaggregation]The aggregate power consumption is
defined as the sum of the푁individual appliance power signal
## 푎
## 1
## (푡),푎
## 2
## (푡), . . .,푎
## 푁
(푡)that run simultaneously plus some noise
휖(푡), accounting for measurement errors. Formally, it is defined as:
## 푥(푡)=
## 푁
## ∑︁
## 푖=0
## 푎
## 푖
## (푡)+휖(푡)(1)
where푥(푡)is the total power consumption measured by the main
meter at timestep푡;푁is the total number of appliances connected
to the smart meter; and휖(푡)is defined as the noise or the measure-
ment error at timestep푡. The NILM challenge relies on accurately
decomposing푥(푡)to retrieve the푎
## 푖
## (푡)components.
## 4  Proposed Approach
The Transformer architecture has demonstrated good performance
when applied to energy disaggregation [1,57,61], but current ap-
proaches do not consider the non-stationary aspect of real-world
smart meter data. We propose NILMFormer as a sequence-to-sequen-
ce Transformer-based architecture designed to handle this phenom-
enon. NILMFormer operates by first stationarizing the input subse-
quence by subtracting its mean and standard deviation. While the
normalized subsequence is passed through a robust convolutional
block that serves as a feature extractor, the removed statistics are
linearly projected in a vector (referred to asTokenStats), and the
proposed TimeRPE module uses the timestamps to compute a po-
sitional encoding matrix. These features are concatenated and fed
into the Transformer block, followed by a simple Head to obtain a
1D sequence of values. The final step consists of linearly project-
ing back theTokenStats(referred to asProjStats) to 2 scalar values
that are then used to denormalize the output, providing the final
individual appliance consumption.
Overall, NILMFormer first splits and encodes separately the
shape, the temporal information, and the intrinsic statistics of the
subsequences, which are then mixed back together in the Trans-
former block. In addition, the output prediction is refined through

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
EmbeddingBlock
TimeRPE
## Subsequence’s
discrete
timestamps
## Transformer Block
z-normalization
## 푑
## 푑
## 4
## 푑
## 푑
## 휎
## Std
## 푑-
## 푑
## 4
## Concatenation
## 휇
## Mean
## Proj휇
## Linear (2x푑)
## Head
denormalization
Input meterreadingsubsequence
## Individualpredictedappliancepower
TokenStats
ProjStats
## ...
Power values
## ...
## ...
## ...
## ...
## Linear(푑x2)
## Proj(휎
## 2
## )
Figure 2: Overview of the NILMFormer architecture.
the linear transformation of the input series statistics, accounting
for the loss of power when disaggregating the signal.
4.1  NILMFormer Architecture
As depicted in Figure 2, NILMFormer results in an encoder that
takes as input a subsequence of an entire household smart meter
reading푥=(풙
## 1
## , ...,풙
## 푛
)and outputs the individual consumption푎=
## (풂
## 1
## , ...,풂
## 푛
)for an appliance. The workflow unfolds in the following
steps.
Step 1: Input subsequence stationarization.The input power
subsequence푥
## 1×푛
is first z-normalized. More specifically, we first
extract the mean and standard deviation of the sequence as휇=
## 1
## 푛
## Í
## 푛
## 푖=0
## 푥
## 푖
and휎=
## √︃
## 1
## 푛
## Í
## 푛
## 푖=0
## (푥
## 푖
## −휇)
## 2
, respectively. Then, we re-
move the extracted mean to each value풙
## 푖
and divide them by the
standard deviation as
## ̃
## 푥=
## 푥−휇
## 휎+휖
, such that the mean of the subse-
quences become 0 and the standard deviation 1.
Step 2: Tokenization.The mean휇and standard deviation휎val-
ues are projected using a learnable linear layer in a푑-dimensional
vector, referred to asTokenStats. In addition, the z-normalized sub-
sequence
## ̃
## 푥
## 1×푛
is passed through the embedding block, used to
extract local features. This block is composed of several convolu-
tional layers using푑−
## 푑
## 4
filters (see Section 4.1.1 for details), which
output a features map푧
## 푑−
## 푑
## 4
## ×푛
. In parallel, the positional encoding
matrix푃퐸
## 푑
## 4
## ×푛
is computed according to the subsequence’s discrete
timestamp information using the TimeRPE module (detailed in
## Section 4.1.3).
Step 3: Features mix.TheTokenStatsand the positional encoding
matrix are concatenated to the extracted subsequence’s features
maps, resulting in a new features map
## ˆ
## 푧
## (푛+1)×푑
. More specifically,
the positional encoding matrix is concatenated along the inner
dimension푑, and theTokenStatsis concatenated along the time
dimension푛(it can be viewed as adding a new token). Then, the
obtained matrix
## ˆ
푧is passed through the Transformer Block (see
Section 4.1.2 for details) that is used to mix the different informa-
tion and learn long-range dependencies. Then, the first token of the
output representation
## ˆ
## 푧
## (푛+1)×푑
, corresponding to theTokenStats, is
removed to obtain a feature map matching the input subsequence
length,푧
## 푛×푑
## . Finally,푧
## 푛×푑
is processed through the output Head,
consisting of a 1D convolutional layer, which maps the latent rep-
resentation back to a 1D series representation,
## ̃
## 푎
## 1×푛
## .
Step 4: Output de-stationarization.TheTokenStatsis projected
back using a learnable linear layer (푑×2) that provides two scalar
values푃푟표푗(휇)and푃푟표푗(휎), referred to asProjStats. These values
are used as a new mean and standard deviation to denormalize the
output
## ̃
## 푎
## 1×푛
and obtain the final prediction as:푎=
## ̃
## 푎∗푃푟표푗(휎)+
푃푟표푗(휇). The subsequence푎is the individual appliance power.
4.1.1  Embedding Block.As depicted in Figure 3(b), the Embedding
Block results in 4 stacked convolutional Residual Units (ResUnit),
each one composed of a convolutional layer, a GeLU activation func-
tion [26], and a BatchNormalization layer [28]. Not that a residual
connection is used between each ResUnit, and a stride parameter
of 1 is employed in each convolutional filter to keep the time di-
mension unchanged. For each Residual Unit푖=1, ...,4, a dilation
parameter푑=2
## 푖
that exponentially increases according to the
ResUnit’s depth is employed. We motivate the choice of using this
feature extractor in Section 5.4.
4.1.2  Transformer Block.The core of NILMFormer relies on a block
composed of푁stacked Transformer layers. Each Transformer layer
is made of the following elements (cf. Figure 3(a)): a normalization
layer, a Multi-Head Diagonally Masked Self-Attention mechanism
(Multi-Head DMSA), a second normalization layer, and a Positional
Feed-Forward Network [58] (PFFN). We use a Multi-Head DMSA
instead of the original Attention Mechanism, as it is more effective
when applied to electricity consumption series analysis [50]. In
addition, we introduce residual connections after the Multi-Head
DMSA and the PFFN, and the use of a Dropout parameter.
4.1.3  Timestamp Related Positional Encoding (TimeRPE).The Trans-
former architecture does not inherently understand sequence order
due to its self-attention mechanisms, which are permutation invari-
ant. Therefore, Positional Encoding (PE) is mandatory to provide
this context, allowing the model to consider the position of each
token in a sequence [58]. Fixed sinusoidal or fully learnable PEs
are commonly used in most current Transformer-based architec-
tures for time series analysis [42], including those proposed for
energy disaggregation [7,61]. This kind of PE consists of adding a
matrix of fixed or learnable weight on the extracted features before
the Transformer block. However, these PEs only help the model

NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-StationarityConference acronym ’XX, June 03–05, 2018, Woodstock, NY
Multi-Head
## DMSA
## ×
## 푁

layer
LayerNorm
LayerNorm
## PFFN
ResUnit 푖=4
Conv1D 푑=2
## 푖
GeLU activation
BatchNorm
ResUnit

## 푖
## =
## 0
## ...
## (a) Embedding Block
## (b) Transformer Block
## 푡
## 0
## 푚
## 푡
## 푖
## 푚
## 푡
## 푛
## 푚
## 푡
## 0
## ℎ
## 푡
## 푖
## ℎ
## 푡
## 푛
## ℎ
## 푡
## 0
## 푑
## 푡
## 푖
## 푑
## 푡
## 푛
## 푑
## 푡
## 0
## 푀
## 푀
## 푖
## 푀
## 푛
## 푡
## 0
## 푡
## 푖
## 푡
## 푛
minutes
## 푡
## 푚
## ∈{0,59}
hours
## 푡
## ℎ
## ∈{0,23}
days
## 푡
## 푑
## ∈{0,6}
## ...
Smart meter  reading subsequence
## ...
Conv1D 푘=1
Positional Encoding matrix
## ...
(c) TimeRPE
## Timestamps
Power values
## 푇
## 푠푖푛
## 푡
## 푚
## =
## 푇
## 푐표푠
## 푡
## 푚
## =
## 푇
## 푠푖푛
## 푡
## ℎ
## =
## 푇
## 푐표푠
## 푡
## ℎ
## =
## 푇
## 푠푖푛
## 푡
## 푑
## =
## 푇
## 푐표푠
## 푡
## 푑
## =
## 푇
## 푠푖푛
## 푡
## 푀
## =
## 푇
## 푐표푠
## 푡
## 푀
## =
months
## 푡
## 푀
## ∈{0,11}
## ...
## ...
## ...
Figure 3: NILMFormer’s architecture parts detail: (a) Embed-
ding Block; (b) Transformer Block; (c) TimeRPE module.
understand local context information (i.e., the given order of the
tokens in the sequence) and do not provide any information about
the global context when operating on subsequences of a longer
series. In the context of NILM, appliance use is often related to
specific periods (e.g., dishwashers running after mealtimes, electric
vehicles charging at night, or on weekends). Moreover, detailed
timestamp information is always available in real-world NILM ap-
plications. Thus, using a PE based on timestamp information can
help the model better understand the recurrent use of appliances.
Timestamp-based PEs have been briefly investigated for time series
forecasting [65] but were always combined with a fixed or learnable
PE and directly added to the extracted features.
Therefore, we proposed the Timestamps Related Positional En-
coding (TimeRPE), a Positional Encoding based only on the dis-
crete timestamp values extracted from the input subsequences. The
TimeRPE module, depicted in Figure 3 (c), takes as input the times-
tamps information푡from the input subsequences, decomposes it
such as minutes푡
## 푚
, hours푡
## ℎ
, days푡
## 푑
, and months푡
## 푀
, and project
them in a sinusoidal basis, as:푇
sin
## (푡
## 푖
## )=sin
## 
## 2휋푡
## 푗
## 푖
## 푝
## 푗
## 
and푇
cos
## (푡
## 푖
## )=
cos
## 
## 2휋푡
## 푗
## 푖
## 푝
## 푗
## 
, with푗∈ {푚,ℎ,푑,푀}and{푝
## 푚
## =59,푝
## ℎ
## =23,푝
## 푑
## =
## 6,푝
## 푀
=11}corresponding to the set of max possible discrete times-
tamp variable. Afterward, the obtained representation is projected
in a higher-dimensional space using a 1D convolution layer with a
kernel of size 1. We evaluate the effectiveness of TimeRPE against
various standard PE used in time series Transformers in Section 5.4.
Table 1: Dataset characteristics and parameters
DatasetsUKDALEREFITEDF 1EDF 2
## Nb. Houses52036924
Avg. recording time223days150days3.7years1year
Sampling rate1min1min10min30min
Max. power limit (휏
## 푚푎푥
## )6000100002400013000
Max. ffill3min3min30min1h30
## Appliances
DishwasherDishwasherCentral HeaterElectric Vehicle
Washing MachineWashing MachineHeatpump
MicrowaveMicrowaveWater Heater
KettleKettleWhiteUsages
## Fridge
## 5  Experimental Evaluation
All experiments are performed on a cluster with 2 Intel Xeon Plat-
inum 8260 CPUs, 384GB RAM, and 4 NVidia Tesla V100 GPUs with
32GB RAM. The source code [46] is in Python v3.10, and the core
of NILMFormer in PyTorch v2.5.1 [45].
## 5.1  Datasets
We use 4 different datasets (see Table 1) that provide the total
power consumed by the house, recorded by a smart meter, and the
individual load power measurement for a set of appliances.
5.1.1  Public Datasets.UKDALE [31] and REFIT [21] are two well-
known datasets used in many research papers to assess the perfor-
mance of NILM approaches [30,54,61,63,66]. The two datasets
contain high-frequency sampled data collected from small groups
of houses in the UK and focus on small appliances.
5.1.2  EDF Datasets.We also use 2 private (EDF) datasets that in-
clude modern appliances not included in the public datasets, such
as Heater systems, Water Heaters and Electric Vehicle chargers.
[EDF Dataset 1]It contains data from 358 houses in France be-
tween 2010 and 2014. Houses were recorded for an average of 1357
days (shortest: 501 days; longest: 1504 days). Houses were moni-
tored with smart meters that recorded the aggregate main power,
as well as the consumption of individual appliances at 10min in-
tervals. Diverse systems, including the central heating system, the
heat pump, and the water heater, have been monitored directly
through the electrical panel. In addition, other appliances, such as
the dishwasher, washing machine, and dryer, were monitored using
meter sockets. The signals collected from these appliances have
been grouped in one channel called "White Appliances".
[EDF Dataset 2]It contains data from 24 houses in France from
July 2022 to February 2024. Data were recorded for an average
of 397 days (shortest: 175 days; longest: 587 days). Houses were
monitored with individual smart meters that recorded the main
aggregate power of the house and a clamp meter that recorded the
power consumed by an electric vehicle’s charger. Aggregate main
power and electric vehicle recharge were sampled 30min intervals.
5.1.3  Data processing.According to the parameters reported in
Table 1, we resampled and readjusted recorded values to round
timestamps by averaging the power consumed during the inter-
valΔ
## 푡
, we forward-filled the missing values, and we clipped the
consumption values between 0 and a maximum power threshold.
In this study, we evaluate the model’s performance based on real-
world scenarios using unseen data from different houses within the

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
Table 2: Overall results for the different disaggregation cases, datasets, and subsequences window length. Each reported result
is the average score obtained for 3 runs. The best score is shown in bold, and the second best is underlined.
ModelNILMFormerBERT4NILMSTNILMBiGRUEnergformerFCNUNet NILMTSILNetBiLSTMDiffNILMDAResNet
Dataset and caseWinMAEMRMAEMRMAEMRMAEMRMAEMRMAEMRMAEMRMAEMRMAEMRMAEMRMAEMR
## UKDALE
## Dishwasher
12822.9    0.53432.20.2733.30.25730.80.32641.40.09639.80.17136.40.2838.70.26736.00.31959.20.01848.00.293
## 256
16.7    0.62634.40.23931.60.28730.90.35940.40.09943.20.14540.60.26848.60.20344.30.27387.60.02266.00.209
51224.1    0.44234.40.26531.20.31427.20.40845.20.09745.30.17147.80.20553.40.19257.40.25595.30.036126.50.085
## Fridge
12831.60.36425.60.47336.90.36436.70.36326.50.47125.80.49323.3    0.53624.60.52824.60.53328.80.49492.90.266
25634.40.29127.00.44736.90.36436.80.36532.30.38427.80.45826.0    0.48533.30.41236.90.36672.60.25633.80.444
51233.30.31526.60.45236.90.36436.70.36433.30.42728.00.47829.80.42537.00.36537.00.36683.90.225122.00.159
## Kettle
1288.70.70511.00.6429.80.6659.80.6713.50.59517.40.51516.20.53518.90.49619.00.51215.30.57394.40.265
## 256
11.00.63512.60.6089.80.66410.00.66410.10.66322.00.43825.80.40623.90.43732.40.36463.10.04638.80.278
5129.80.66613.60.58310.80.63211.60.61812.30.60229.20.35343.70.22334.60.31460.40.12688.30.037113.80.068
## Microwave
12810.10.1448.70.228.90.18512.80.0889.00.20114.60.11613.80.06215.90.05115.60.04516.70.01123.50.044
## 256
10.10.139.10.1711.20.09914.30.04311.00.14915.20.07514.30.0416.30.03216.10.02837.30.0129.10.036
5128.30.18710.40.11110.90.06612.40.0410.50.12917.00.02714.70.01816.00.02215.90.01256.00.00980.60.013
## Washer
12812.4    0.24716.40.13721.40.11420.60.10929.40.13624.90.11422.70.10928.50.0628.60.06834.80.02135.90.075
2568.50.35826.30.08426.60.08119.00.09332.00.11632.90.08925.40.09334.50.05132.90.05550.10.01449.20.042
51210.8    0.26831.50.0926.50.09821.10.08631.40.08638.30.05829.20.06634.80.05239.30.04466.10.01390.50.019
## REFIT
## Dishwasher
12829.1    0.33249.30.08141.10.13644.50.16935.80.08239.30.21849.60.16647.00.13470.00.09169.00.018242.20.044
25645.50.31244.20.14540.50.14970.40.06954.90.08450.20.15159.20.09369.40.06572.70.05774.30.02125.70.049
## 512
42.7    0.20556.00.11746.30.12175.90.07264.60.11252.80.13459.10.08672.60.04690.20.03783.10.022231.90.042
## Kettle
## 128
9.30.52216.90.4378.80.5299.40.50812.40.42615.50.38715.70.36817.60.34818.00.3533.80.00953.40.121
## 256
12.50.4549.20.5049.80.50310.00.49616.00.35918.20.33219.10.3118.90.31225.40.24934.10.14154.90.123
5128.60.52610.60.46211.00.43919.30.32917.70.30622.50.25928.30.20526.20.23342.50.05443.20.014119.90.051
## Microwave
1285.70.1110.70.07411.90.049.10.0410.00.04612.90.0512.90.04310.10.02210.20.02322.20.00643.40.014
2569.70.0510.70.0569.70.02910.00.02310.20.03210.30.02813.80.0239.80.01911.10.01721.70.00777.60.009
## 512
6.40.08210.70.0298.00.0249.70.0159.90.02711.40.01511.30.0139.50.0149.20.00731.00.006540.20.007
## Washer
12818.90.2533.60.17233.00.12630.60.11536.90.11535.50.11242.70.135.80.10940.90.09748.40.023101.40.048
25622.3    0.17234.80.10930.90.12534.00.130.70.10942.00.09842.40.0742.80.07740.00.07657.70.026166.50.029
## 512
20.6    0.16831.00.0929.00.1137.40.07532.80.09740.10.09435.70.08640.50.06943.00.06350.30.027188.40.029
## EDF1
## Heater
128283.4    0.52312.10.457310.30.44342.70.435354.00.445328.00.428352.80.437374.60.416345.00.425351.60.434529.70.305
256263.6   0.525297.20.458290.80.456323.00.425371.90.414316.60.437359.00.402337.70.405330.50.415333.50.442533.60.296
512255.8   0.521283.30.456275.30.456321.10.411368.70.386307.90.424323.10.403314.60.396330.00.395389.20.336497.20.26
## Heatpump
128270.5   0.548288.50.504283.30.523282.00.526274.90.53328.10.475362.90.441326.60.462337.40.446344.00.452507.40.356
256276.2   0.545292.00.504282.20.501288.10.507280.40.51353.10.43364.00.422345.60.451370.90.412304.90.486542.60.272
512258.1   0.533295.90.475290.30.489303.00.457299.20.48392.70.374349.90.394363.30.388377.50.375362.90.397579.70.272
WaterHeater
12888.2    0.686113.10.613134.90.566112.50.619135.60.551185.20.46224.90.364169.80.494178.90.478131.00.55268.80.361
25691.8    0.676126.30.591136.40.564114.20.614146.50.521207.60.428238.90.337203.60.443223.70.4104.80.639342.20.255
## 512
92.4    0.668111.20.615138.10.55118.50.601181.00.464223.70.392237.90.34236.90.372274.80.319126.70.587356.00.256
## White Appl.
12890.8    0.209105.20.203111.60.17199.30.209109.80.179120.90.158116.90.139127.50.123128.30.145134.10.083164.50.105
25686.1    0.231104.20.18999.30.191106.00.176114.80.133130.40.143119.80.144131.30.107133.10.119126.50.069206.80.098
51279.5    0.21594.20.17889.70.178102.80.128107.90.123114.80.103115.20.099116.80.102124.00.088203.20.06192.20.071
## EDF2
## EV
128102.5   0.617169.20.472109.90.57210.40.402134.80.507189.50.428245.70.384242.50.372238.40.371286.30.322434.50.233
256111.8   0.582168.60.456119.10.553227.80.36180.10.396232.20.381262.50.31299.40.303301.20.313213.80.404932.80.097
512114.3   0.616152.50.544130.60.597327.10.304147.30.556323.20.326398.40.205357.70.206481.40.122295.30.3541897.10.046
Avg. Score Per Metrics70.70.484.50.32881.20.326394.50.304293.20.2914107.80.2611116.00.2412114.50.2373122.50.2216122.20.1838261.30.1462
Avg. Rank Per Metrics1.762   1.9053.6673.4763.313.7624.2624.6674.8814.7146.2865.7146.8576.817.2627.698.1438.1438.8339.23810.7389.881
## Avg. Rank1.8333.5363.5714.4644.7986.06.8337.4768.1439.03610.31
same dataset [52]. Distinct houses were used for training and eval-
uation to ensure robust performance assessment. For the UKDALE
dataset, we utilized houses 1, 3, 4, and 5 for training and house 2
for testing. This selection was made because only houses 1 and 2
contained all the appliances. Note that for UKDALE, 80% of the
data from house 1 was used for training, while 20% was used as a
validation set to prevent overfitting.
For the REFIT and EDF1 datasets, which contain more houses, 2
houses were reserved for testing, 1 house for validation, and the
remaining houses were used for training. For EDF2, which contains
more than 350 houses, 70% of the houses were used for training,
10% for validation, and the remaining 20% for evaluation.
For training and evaluating the model, the entire recorded con-
sumption data of the different houses was sliced into subsequences
using a non-overlapping window of length푤. Subsequences con-
taining any remaining missing values were discarded. To ensure
training stability, we scaled the data to a range between 0 and 1 by
dividing the consumption values (both aggregate and individual
appliance power) by the maximum power threshold휏
## 푚푎푥
for each
dataset as reported in Table 1. Consequently, before evaluating the
models, we denormalized the data by multiplying it by휏
## 푚푎푥
## .
## 5.2  Evaluation Pipeline
We compare our solution against several SotA sequence-to-sequence
NILM solutions. We include two recurrent-based architectures, BiL-
STM [30] and BiGRU [51] that combine Convolution and Recur-
rent layers; three convolutional-based baselines, FCN [63], UNet-
NILM [53] and DAResNet [60]; as well as a recent diffusion-based
proposed approach, DiffNILM [54]. In addition, we include 4 Tranfor-
mer-based baselines, BERT4NILM [61], Energformer [1], STNILM [57],
nd TSILNet [66], which integrates both Transformer and recurrent
layers. We used the default parameters provided by the authors and
trained all the models using the Mean Squared Error loss.
5.2.1  Window length sensitivity.We evaluate each baseline using
different subsequences windows length푤to assess the sensitivity
of the results on this parameter. We experimented with windows
length푤={128,256,512}to standardize results across datasets.
We also considered using window lengths corresponding to specific
periods based on the sampling rate but observed no significant
changes in the results.
5.2.2  Evaluation metrics.We  assess  the  energy  disaggregation
quality  performance  of  the  different  baselines  using  2  metrics.
For each metric,푇represents the total number of intervals while
## 푦
## 푡
is the true and
## ˆ
## 푦
## 푡
is the predicted power usage of an appli-
ance. The first metric is the standard Mean Absolute Error (MAE):
## 푀퐴퐸=
## 1
## 푇
## Í
## 푇
## 푡=1
## |
## ˆ
## 푦
## 푡
## −푦
## 푡
|. The second metric is the Matching Ratio

NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-StationarityConference acronym ’XX, June 03–05, 2018, Woodstock, NY
NILMFormerBERT4NILMBiGRU
## FCN
UNet-NILM
## Energformer
## STNILM
## Time
## Time
## Time
## Time
## Time
## Time
## Time
Household meter reading
True appliance power
Models prediction
kW
Figure 4: Qualitative disaggregation results for Electric Vehicle (EDF1) on a sample example for the 7 best baselines. (푤=128).
(MR), based on the overlapping rate of true and estimated predic-
tion, and stated to be the best overall indicator performance [39]:
## 푀푅=
## Í
## 푁
## 푡=1
## 푚푖푛(
## ˆ
## 푦
## 푡
## ,푦
## 푡
## )
## Í
## 푁
## 푡=1
## 푚푎푥(
## ˆ
## 푦
## 푡
## ,푦
## 푡
## )
## .
## 5.3  Results
Table 2 lists the results for the 4 datasets and each different appliance
energy disaggregation case and subsequence window length. First,
we note that NILMFormer outperforms other solutions overall for
the 2 metrics (avg. rank per metric and avg. total Rank). More specif-
ically, we note an improvement of over 15% in terms of MAE and
22% in MR (on average across all the datasets and cases) compared
to the second-best model for each metric, STNILM and BERT4NILM,
respectively. NILMFormer is outperformed only on the Fridge dis-
aggregation case on the UKDALE dataset by UNet-NILM. This is
because the fridge is always ON and present in all houses. As a
result, the scheme adopted to address the non-stationary nature of
the data caused the model to struggle when isolating the constant,
baseline consumption that the fridge represents. We also provide
in Figure 4 an example of the disaggregation quality of the 7 best
baselines on the Electric Vehicle case (EDF1).
5.4  Impact of Design Choices on Performance
We perform a detailed evaluation to assess the performance of
different key parts of NILMFormer. We first study the effectiveness
of the two mechanisms proposed to mitigate the non-stationary
aspect of the data (i.e., theTokenStatsand theProjStats). Then, we
assess the proposed TimeRPE’s performance against other PEs
usually used in Transformers for time series analysis.
We utilize a Critical Difference Diagram (CD-Diagram) to com-
pare the performance, computed based on the rank of the different
variants. This method, proposed in [3], involves performing a Fried-
man test followed by a post-hoc Wilcoxon test on the calculated
ranks, according to significance level훼=0.1. The rank is given
by computing an overall score according to the two disaggrega-
tion metrics (MAE and MR) averaged across all the datasets, cases,
and window lengths, previously normalized to the same range (be-
tween 0 and 1). In addition, we note that bold lines in CD-diagrams
indicate insignificant differences between the connected methods.
5.4.1  Non-stationarity consideration.We assess the influence of
the two mechanisms adopted to mitigate the non-stationary as-
pect of the data. More precisely, we compared the following op-
tions: (None): not considering any mechanism to handle the non-
stationary aspect of the data (as the previously proposed NILM
approach); (RevIN): applying the RevIN [33] per subsequences nor-
malization/denormalization scheme to the input/output without
propagating any information inside the network, i.e., the approach
currently adopted in most of TSF architecture [27,42]; (w/ To-
kenStats): propagating only theTokenStatsinside the Transformer
block, i.e., the extracted input statistics휇and휎are re-used to
denormalize the output, (w/ ProjStats): using only the learnable
projection of the extracted input statistics휇and휎to denormalize
the output without propagating information in the Transformer
Block; and, (w/ TokenStats and w/ ProjStats); the final proposed
mechanism include in NILMFormer. The results (cf. Figure 5(a))
demonstrate that applying only the RevIN scheme leads to worse
performance than using the proposed architecture without any
mitigation of the non-stationarity effect and, thus, confirms that
level information is crucial for NILM. In addition, omitting either
TokenStatsorProjStatsresults in a performance drop, confirming
the essential role of both.
5.4.2  Positional Encoding.We evaluate the effectiveness of the
proposed TimeRPE against usual standard PE methods used in the
NILM literature. More specifically, we investigate the following
options: (1) removing the PE (NoPE); (2) concatenating or adding
the TimeRPE’s PE matrix to the embedding features; (3) replac-
ing TimeRPE by (Fixed) the original PE proposed for Transformer
in [58]); (tAPE) a fixed PE proposed recently, especially for time
series Transformer [22]; (Learnable) a fully learnable one, as pro-
posed for time series analysis in [62], and used in the NILM lit-
erature [55,59,61]. The critical diagram in Figure 5(b) shows the
significant superiority of TimeRPE over other PE options. Moreover,
we noticed that concatenating the PE information with the features
extracted from the Embedding Block instead of adding it lead to
significantly better performance. We assume that concatenating
the PE instead of adding lead helps the model to differentiate the
two pieces of information.
5.4.3  Embedding Block.We evaluate the impact of different em-
bedding blocks for extracting features from the aggregate power
signal. More specifically, we investigate replacing the proposed
Dilated Residual Embedding Block by: (Linear) a simple linear
embedding layer that maps each time step of the input sequence
model; (Patchify) a patching embedding, an approach used by nu-
merous Transformer for time series analysis [42,59], that involves
dividing the input series in patches (i.e., subsequences); using a
convolutional layer with stride and kernel of the patch length; and
(ResBlock) a simple Residual Convolution Block without dilation
(with k=3). The results (cf. Figure 5(c)) demonstrates the necessity

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
(b) NILMFormer with Different Positional Encoding
(a) NILMFormer with Different Non-stationary Mechanisms
w/ TokenStats
and w/ ProjStats
w/ TokenStats
w/ ReVIN
## None
w/ ProjStats
## 4.88
## 3.43
## 2.52
## 1.71
## 2.45
## 4.48
## 4.38
## 3.76
## 2.21
## 2.69
## 3.47
w/ TimeRPE (concat.)
w/ TimeRPE (added)
w/ tAPE
No PE
w/ Learnable PE
w/ Fixed PE
w/ Dilated ResBlock
w/ Simple ResBlock
w/ Patchify
w/ Linear
(c) NILMFormer with Different Embedding Block
## 1.63
## 1.87
## 3.8
## 2.7
Figure 5: CD-diagram of the average rank (avg. of the MAE
and MR across all datasets, cases, and window lengths) to
study the impact of: (a) the proposed mechanisms to mitigate
the non-stationary aspect of the data, (b) different PE, and
(c) different embedding block.
## Overall

metrics

score
## 풅
## 풎풐풅풆풍
## ퟖ
## 풅
## 풎풐풅풆풍
## ퟒ
## 풅
## 풎풐풅풆풍
## ퟐ
% of PE information
Figure 6: Overall metrics score (avg. of the MAE and MR
for푤=256, over all the datasets and cases) by varying the
ratio of positional encoded information (given by TimeRPE)
according to the inner model dimension (푑
## 푚표푑푒푙
## =128).
of the convolution layers to extract localized feature patterns. Using
the proposed Embedding Block leads to a slight (but not signifi-
cant) increase compared to the simple ConvBlock. Note that using
the patch embedding leads to the worst results, suggesting that
this type of embedding does not suit sequence-to-sequence NILM
solutions.
5.4.4  Impact of Positional Encoding Ratio on Model Performance.
As detailed in Section 4, NILMFormer concatenates the positional-
encoding matrix produced by TimeRPE with the feature vectors gen-
erated by the Embedding Block along the inner model dimension푑.
To quantify how much representational budget should be allocated
to positional cues, we train NILMFormer while varying the share
of channels reserved for TimeRPE (using window size푤=256
and푑=128). Specifically, we examine three ratios,
## 
## 푑
## 8
## ,
## 푑
## 4
## ,
## 푑
## 2

, cor-
responding to dedicating12.5%,25%, and50%of the channels to
positional information.
The results in Figure 6 reveal that allocating one quarter of
the channels to positional encoding (
## 푑
## 4
) offers the best trade-off,
delivering the highest disaggregation accuracy. Increasing the share
to
## 푑
## 2
brings no further gains, while reducing it to
## 푑
## 8
only slightly
affects performance.
## 6  Deployed Solution
Since 2015, EDF has offered to its clients a consumption monitoring
service calledMon Suivi Conso[13], enabling clients to track their
consumption. The service is accessible via the web and a mobile
app. A new feature was released in 2018 to provide the clients with
an estimation of their annual individual appliance consumption.
The backbone algorithm relied on semi-supervised statistical meth-
ods that used the customers’ static information [14,15]. However,
recent user feedback indicated a growing demand for more granular
and personalized insights, consumption estimates down to daily fre-
quency, and percentage-based cost breakdowns per appliance [11].
In response, EDF recently explored the use of TSER [56] to infer
monthly appliance-level consumption. These approaches yielded
an improvement over the original semi-supervised methods em-
ployed inMon Suivi Conso, reducing the mean absolute error (MAE)
of monthly misassigned consumption across various appliances
(e.g., heaters, water heaters, and electric vehicles). Despite these ad-
vances, monthly-level feedback remains relatively coarse, limiting
its practical value for end-users. Consequently, NILM-based algo-
rithms, which can disaggregate consumption at a per-timestamp
level, offer a promising alternative for replacing and enhancing the
current individual appliance consumption feedback feature.
[NILMFormer Applied to Daily and Monthly Feedback]Em-
ploying NILMFormer for per-period feedback is straightforward, as
the smart meters in France collect data at half-hour intervals. Thus,
to provide the individual appliance consumption over a period of
time, we adapted NILMFormer as follows:
- The electricity consumption series of a client is sliced into subse-
quences using a tumbling (non-overlapping) window of size푤.
- Each subsequence is then passed to a NILMFormer instance
trained to disaggregate the signal for a specific appliance푎.
- The predictions are then concatenated.
- Finally, the appliance predicted power consumption is returned,
summing over the period of interest (day, week, or month).
[Comparison to TSER Approaches]We experimentally evaluate
the performance of our approach against the 5 best SotA TSER
baselines reported in [56], including XGBoost [6], Rocket [9] (a ran-
dom convolutional-based regressor), and 3 convolutional-based DL
regressors (ConvNet, ResNet, and InceptionTime). We consider two
settings: predicting the per-day and per-month appliance consump-
tion. To train the TSER baselines, we preprocessed the datasets
(EDF1 and EDF2) by slicing the entire consumption into subse-
quences (day or month), and by summing the total consumption
of the appliance over that period to obtain the label. Note that all
baselines were initialized using default parameters, and DL-based
methods were trained using the Mean Squared Error Loss.
To evaluate our approach, we reused the different instances
of NILMFormer trained on timestamp energy disaggregation in
Section 5.3 and applied the framework described above.

NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-StationarityConference acronym ’XX, June 03–05, 2018, Woodstock, NY






NILMFormer
## Rocket
## Inception
ConvNet
xgboost
ResNet
## EV
## Water Heater
## Heater
## Heatpump
## White Appliances
## 퐌퐀퐄

(kW)







Monthly consumption
in january 2025
Electricity consumption feedback
47% of your total
electricity consumption
## ...
435 kWh
Daily consumption
(a) Mean Average Error (MAE) in kW for different approaches applied to daily (left) and monthly
(right) consumption restitution of five different appliances.
(b) User interface of EDF’s Mon Suivi Conso
mobile application.
monthly electricity
consumption
## Heater
Figure 7: (a) Results comparison for per-period energy estimation; (b) Example of feedback available to a client through the
user interface of EDF’sMon Suivi Consomobile application.
Results.The results (cf. Figure 7(a)) demonstrate the superiority
of NILMFormer, which significantly outperforms TSER baselines
applied to daily and monthly appliance consumption prediction. On
average, across the five appliances evaluated, NILMFormer achieves
a 51% reduction in MAE for daily consumption compared to the
second-best baseline (XGBoost), and a 151% reduction in MAE for
monthly consumption compared to the second-best baseline (In-
ception). These findings highlight the effectiveness of our approach
for deployment inMon Suivi Conso, providing substantially more
accurate feedback to clients than previous approaches.
6.1  Deployment Insights and Performance
DATANUMIA, an EDF subsidiary, oversees the deployment and in-
tegration of NILMFormer intoMon Suivi Conso, gradually replacing
the existing solution for individual appliance consumption feed-
back. The final delivered algorithm uses a separate model for each
appliance, enabling the solution to predict consumption for each
appliance independently. These individual predictions are then ag-
gregated to produce a final output showing the percentage of each
appliance’s consumption relative to the total (see Figure 7(b)). Users
can also track their electricity usage according to different time in-
tervals: daily, monthly, and annual periods, both in kilowatt-hours
and euros (based on their contracted rates).
[Infrastructure and Performance]The deployed solution is
hosted on Amazon Elastic Compute Cloud (EC2) using 6m6i.8xlarge
instances, each providing 32 vCPUs. The individual appliance con-
sumption estimation runs weekly on the entire customer base that
has consented to data analysis—around 3.6 million customers. Post-
launch performance evaluations shows that the system currently
handles 100 customer requests per second, processing the entire
dataset in approximately 11 hours.
[User Engagement]During the last quarter of 2024, operational
statistics indicate 7 million visits to the consumption monitoring
solution that uses our approach, with 2.2 million visits to the indi-
vidual appliance consumption feedback through the web interface.
For the mobile application, out of 7 million visits to this application,
6.2 million visits focused on the per-appliance consumption feature,
demonstrating considerable interest in the information that our
solution provides.
[Decision Making]EDF recently applied this solution to a subset
of consenting consumers on La Réunion [16], focusing on elec-
tric vehicle (EV) charging habits, and the impact of different off-
peak charging systems (e.g., smart plugs versus dedicated charging
stations) on overall consumption. Through NILMFormer, it was
possible to pinpoint exact EV charging times and calculate the asso-
ciated total consumption. The results showed notable benefits for
customers using controlled charging stations, offering improved
insights into peak/off-peak ratios, and demonstrating the value of
NILMFormer in enhancing data-driven decision-making for EDF.
## 7  Conclusions
We proposed NILMFormer, a DL sequence-to-sequence Transformer
for energy disaggregation, designed to address the data distribu-
tion drift problem that occurs when operating on subsequences of
consumption series. NILMFormer employs a stationarization/de-
stationarization scheme tailored to the NILM problem and uses
TimeRPE, a novel PE based only on the subsequence’s timestamp
information. The results show that NILMFormer significantly out-
performs current SotA NILM solutions on 4 different datasets. NILM-
Former outperforms the previous method for individual appliance
per period feedback. Our solution has been successfully deployed as
the backbone algorithm for EDF’s consumption monitoring service,
delivering detailed insights to millions of customers.
## Acknowledgments
Supported by EDF R&D, ANRT French program, and EU Horizon
projects AI4Europe (101070000), TwinODIS (101160009), ARMADA
(101168951), DataGEMS (101188416), RECITALS (101168490), and
by푌Π퐴퐼Θ퐴& NextGenerationEU project HARSH (푌Π3푇퐴−0560901).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
## References
[1]Georgios F. Angelis, Christos Timplalexis, Athanasios I. Salamanis, Stelios Krini-
dis, Dimosthenis Ioannidis, Dionysios Kehagias, and Dimitrios Tzovaras. 2023. En-
ergformer: A New Transformer Model for Energy Disaggregation.IEEE Transac-
tions on Consumer Electronics69, 3 (2023), 308–320. doi:10.1109/TCE.2023.3237862
[2]K. Carrie Armel, Abhay Gupta, Gireesh Shrimali, and Adrian Albert. 2013.  Is
disaggregation the holy grail of energy efficiency? The case of electricity.Energy
Policy52 (2013), 213–234. doi:10.1016/j.enpol.2012.08.062
[3]Alessio Benavoli, Giorgio Corani, and Francesca Mangili. 2016. Should We Really
Use Post-Hoc Tests Based on Mean-Ranks?Journal of Machine Learning Research
17, 5 (2016), 1–10.  http://jmlr.org/papers/v17/benavoli16a.html
[4]Laurent Bozzi and Philippe Charpentier. 2018. Évaluation d’Impact sur la Con-
sommation Électrique de la Solution Digitale e.quilibre d’EDF. InJournées de
Statistique (JdS). Société Française de Statistique (SFdS), France.
## [5]
Xiaomin Chang, Wei Li, Chunqiu Xia, Qiang Yang, Jin Ma, Ting Yang, and Albert Y.
Zomaya. 2022. Transferable Tree-Based Ensemble Model for Non-Intrusive Load
Monitoring.IEEE Transactions on Sustainable Computing7, 4 (2022), 970–981.
doi:10.1109/TSUSC.2022.3175941
## [6]
Tianqi Chen and Carlos Guestrin. 2016.  XGBoost: A Scalable Tree Boosting
System. InProceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining(San Francisco, California, USA)(KDD
’16). ACM, New York, NY, USA, 785–794.  doi:10.1145/2939672.2939785
[7]Xu Cheng, Meng Zhao, Jianhua Zhang, Jinghao Wang, Xueping Pan, and Xiufeng
Liu. 2022.  TransNILM: A Transformer-based Deep Learning Model for Non-
intrusive Load Monitoring. InProceedings of the 2022 International Conference on
High Performance Big Data and Intelligent Systems (HDIS). 13–20.  doi:10.1109/
## HDIS56859.2022.9991439
[8]Stanislav Chren, Bruno Rossi, and Tomáš Pitner. 2016. Smart grids deployments
within EU projects: The role of smart meters. In2016 Smart Cities Symposium
Prague (SCSP). 1–5. doi:10.1109/SCSP.2016.7501033
[9]Angus Dempster, François Petitjean, and Geoffrey I. Webb. 2019. ROCKET: Excep-
tionally fast and accurate time series classification using random convolutional
kernels.CoRRabs/1910.13051 (2019). arXiv:1910.13051  http://arxiv.org/abs/1910.
## 13051
[10]  DMLC. 2024. eXtreme Gradient Boosting.  https://github.com/dmlc/xgboost
[11]  EDF. 2024. Private communication with DATANUMIA Team Manager.
## [12]
EDF. 2025. The digital revolution driving energy efficiency.  https://www.edf.fr
## /en/the-edf-group/taking-action-as-a-responsible-company/corporate-social-
responsibility/the-digital-revolution-driving-energy-ef ficiency
[13]EDF. 2025. Solution suivi conso EDF.  https://particulier.edf.fr/f r/accueil/bilan-
consommation/solution-suivi-conso.html
## [14]
Gregory Yard EDF, Laurent Bozzi. French Patent FR1451531, 2014. ESTIMATION
## DE LA CONSOMMATION ELECTRIQUE D’UN EQUIPEMENT DONNE PARMI
UN ENSEMBLE D’EQUIPEMENTS ELECTRIQUES.  https://data.inpi.fr/brevets/F
## R1451531
## [15]
Melanie  Cazes  EDF,  Laurent  Bozzi.  French  Patent  FR3017975,  2016.    ESTI-
## MATION  FINE  DE  CONSOMMATION  ELECTRIQUE  POUR  DES  BESOINS
DE CHAUFFAGE/CLIMATISATION D’UN LOCAL D’HABITATION.https:
//data.inpi.fr/brevets/FR3017975
[16]EDF à la Réunion. 2024. Sarz la Kaz. https://reunion.edf.fr/edf -a-la-reunion/actu
alites-a-la-reunion/sarz-la-kaz.  Accessed: 2025-02-07.
[17]  Anthony Faustine and Lucas Pereira. 2020. Multi-Label Learning for Appliance
Recognition in NILM Using Fryze-Current Decomposition and Convolutional
Neural Network.Energies13, 16 (2020).  doi:10.3390/en13164154
## [18]
Anthony Faustine, Lucas Pereira, Hafsa Bousbiat, and Shridhar Kulkarni. 2020.
UNet-NILM: A Deep Neural Network for Multi-tasks Appliances State Detection
and Power Estimation in NILM. InProceedings of the 5th International Workshop on
Non-Intrusive Load Monitoring(Virtual Event, Japan)(NILM’20). Association for
Computing Machinery, New York, NY, USA, 84–88.  doi:10.1145/3427771.3427859
[19]Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-
Alain Muller, and François Petitjean. 2020. InceptionTime: Finding AlexNet for
time series classification.Data Mining and Knowledge Discovery34, 6 (sep 2020),
1936–1962. doi:10.1007/s10618-020-00710-y
[20]William Fedus, Barret Zoph, and Noam Shazeer. 2022.   Switch transformers:
scaling to trillion parameter models with simple and efficient sparsity.J. Mach.
Learn. Res.23, 1, Article 120 (Jan. 2022), 39 pages.
[21]Steven Firth, Tom Kane, Vanda Dimitriou, Tarek Hassan, Farid Fouchal, Michael
Coleman, and Lynda Webb. 2017. REFIT Smart Home dataset. (6 2017). doi:10.
## 17028/rd.lboro.2070091.v1
[22]Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I Webb, and Mahsa Salehi.
- Improving position encoding of transformers for multivariate time series
classification.Data Mining and Knowledge Discovery38, 1 (2024), 22–48.
[23]Daisy H. Green, Aaron W. Langham, Rebecca A. Agustin, Devin W. Quinn, and
Steven B. Leeb. 2023. Adaptation for Automated Drift Detection in Electrome-
chanical Machine Monitoring.IEEE Transactions on Neural Networks and Learning
Systems34, 10 (2023), 6768–6782. doi:10.1109/TNNLS.2022.3184011
[24]G.W. Hart. 1992. Nonintrusive appliance load monitoring.Proc. IEEE80, 12 (1992),
1870–1891. doi:10.1109/5.192069
[25]Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual
Learning for Image Recognition. doi:10.48550/ARXIV.1512.03385
[26]Dan Hendrycks and Kevin Gimpel. 2016. Gaussian Error Linear Units (GELUs).
doi:10.48550/ARXIV.1606.08415
## [27]
## Romain Ilbert, Ambroise Odonnat, Vasilii Feofanov, Aladin Virmaux, Giuseppe
Paolo, Themis Palpanas, and Ievgen Redko. 2024.  SAMformer: Unlocking the
Potential of Transformers in Time Series Forecasting with Sharpness-Aware
Minimization and Channel-Wise Attention. arXiv:2402.10198 [cs.LG]
[28]Sergey Ioffe and Christian Szegedy. 2015.  Batch Normalization: Accelerating
Deep Network Training by Reducing Internal Covariate Shift. doi:10.48550/AR
## XIV.1502.03167
[29]Maria Kaselimi, Eftychios Protopapadakis, Athanasios Voulodimos, Nikolaos
Doulamis, and Anastasios Doulamis. 2022. Towards Trustworthy Energy Disag-
gregation: A Review of Challenges, Methods, and Perspectives for Non-Intrusive
Load Monitoring.Sensors22 (08 2022), 5872. doi:10.3390/s22155872
## [30]
Jack Kelly and William Knottenbelt. 2015.  Neural NILM. InProceedings of the
2nd ACM International Conference on Embedded Systems for Energy-Efficient Built
Environments. ACM. doi:10.1145/2821650.2821672
[31]Jack Kelly and William Knottenbelt. 2015.   The UK-DALE dataset, domestic
appliance-level electricity demand and whole-house demand from five UK homes.
Scientific Data2 (03 2015). doi:10.1038/sdata.2015.7
[32]Hyungsul Kim, Manish Marwah, Martin F. Arlitt, Geoff Lyon, and Jiawei Han.
- Unsupervised Disaggregation of Low Frequency Power Measurements. In
SDM.  https://api.semanticscholar.org/CorpusID:18447017
[33]Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and
Jaegul Choo. 2021. Reversible Instance Normalization for Accurate Time-Series
Forecasting against Distribution Shift. InInternational Conference on Learning
Representations.  https://openreview.net/f orum?id=cGDAkQo1C0p
[34]J. Zico Kolter. 2011.  REDD : A Public Data Set for Energy Disaggregation Re-
search.
[35]Pauline Laviron, Xueqi Dai, Bérénice Huquet, and Themis Palpanas. 2021. Electric-
ity Demand Activation Extraction: From Known to Unknown Signatures, Using
Similarity Search. Ine-Energy ’21: The Twelfth ACM International Conference on
Future Energy Systems, Virtual Event, Torino, Italy, 28 June - 2 July, 2021, Herman
de Meer and Michela Meo (Eds.). ACM, 148–159. doi:10.1145/3447555.3464865
[36]Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. 2022. Non-stationary
Transformers: Exploring the Stationarity in Time Series Forecasting. InNeu-
ral Information Processing Systems.  https://api.semanticscholar.org/CorpusID:
## 252968420
[37]Prajowal Manandhar, Hasan Rafiq, Edwin Rodriguez-Ubinas, and Themis Pal-
panas. 2024. New Forecasting Metrics Evaluated in Prophet, Random Forest, and
Long Short-Term Memory Models for Load Forecasting.Energies17, 23 (2024).
doi:10.3390/en17236131
## [38]
Luca Massidda, Marino Marrocu, and Simone Manca. 2020. Non-Intrusive Load
Disaggregation by Convolutional Neural Network and Multilabel Classification.
Applied Sciences10, 4 (2020).  doi:10.3390/app10041454
## [39]
Ebony Mayhorn, Greg Sullivan, Joseph M. Petersen, Ryan Butner, and Erica M.
Johnson. 2016. Load Disaggregation Technologies: Real World and Laboratory
Performance.  https://api.semanticscholar.org/CorpusID:115779193
[40]Matthew Middlehurst, Ali Ismail-Fawaz, Antoine Guillaume, Christopher Holder,
## David Guijo Rubio, Guzal Bulatova, Leonidas Tsaprounis, Lukasz Mentel, Martin
Walter, Patrick Schäfer, and Anthony Bagnall. 2024. aeon: a Python toolkit for
learning from time series. arXiv:2406.14231 [cs.LG]  https://arxiv.org/abs/2406.
## 14231
[41]Megan Milam and G. Kumar Venayagamoorthy. 2014. Smart meter deployment:
US initiatives. InISGT 2014. 1–5. doi:10.1109/ISGT.2014.6816507
[42]Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023.
A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In
International Conference on Learning Representations.
[43]Eduardo Ogasawara, Leonardo C. Martinez, Daniel de Oliveira, Geraldo Zimbrão,
Gisele Lobo Pappa, and Marta Mattoso. 2010. Adaptive Normalization: A novel
data normalization approach for non-stationary time series.The 2010 International
Joint Conference on Neural Networks (IJCNN)(2010), 1–8.  https://api.semanticsc
holar.org/CorpusID:5757527
[44]Keiron O’Shea and Ryan Nash. 2015. An Introduction to Convolutional Neural
Networks.CoRRabs/1511.08458 (2015). arXiv:1511.08458  http://arxiv.org/abs/
## 1511.08458
## [45]
## Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
## Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan
Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
Chintala. 2019.PyTorch: an imperative style, high-performance deep learning
library. Curran Associates Inc., Red Hook, NY, USA.
[46]Adrien Petralia. 2025. Source code of NILMFormer experiments.  https://github
.com/adrienpetralia/NILMFormer
[47]Adrien Petralia, Paul Boniol, Philippe Charpentier, and Themis Palpanas. 2025.
DeviceScope: An Interactive App to Detect and Localize Appliance Patterns in

NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-StationarityConference acronym ’XX, June 03–05, 2018, Woodstock, NY
Electricity Consumption Time Series . In2025 IEEE 41st International Conference
on Data Engineering (ICDE). 4552–4555.  doi:10.1109/ICDE65448.2025.00350
[48]Adrien Petralia, Paul Boniol, Philippe Charpentier, and Themis Palpanas. 2025.
Few Labels are All You Need: A Weakly Supervised Framework for Appliance
Localization in Smart-Meter Series . In2025 IEEE 41st International Conference on
Data Engineering (ICDE). 4386–4399.  doi:10.1109/ICDE65448.2025.00329
## [49]
Adrien Petralia, Philippe Charpentier, Paul Boniol, and Themis Palpanas. 2023.
Appliance Detection Using Very Low-Frequency Smart Meter Time Series. In
Proceedings of the 14th ACM International Conference on Future Energy Systems
(Orlando, FL, USA)(e-Energy ’23). Association for Computing Machinery, New
York, NY, USA, 214–225.  doi:10.1145/3575813.3595198
[50]Adrien  Petralia,  Philippe  Charpentier,  and  Themis  Palpanas.  2023.   ADF  &
TransApp: A Transformer-Based Framework for Appliance Detection Using
Smart Meter Consumption Series.Proc. VLDB Endow.17, 3 (nov 2023), 553–562.
doi:10.14778/3632093.3632115
[51]Daniel Precioso Garcelán and David Gomez-Ullate. 2023. Thresholding methods
in non-intrusive load monitoring.The Journal of Supercomputing79 (04 2023),
1–24. doi:10.1007/s11227-023-05149-8
[52]Hasan  Rafiq,  Prajowal  Manandhar,  Edwin  Rodriguez-Ubinas,  Omer  Ahmed
Qureshi, and Themis Palpanas. 2024.  A review of current methods and chal-
lenges of advanced deep learning-based non-intrusive load monitoring (NILM)
in residential context.Energy and Buildings305 (2024), 113890.  doi:10.1016/j.en
build.2024.113890
[53]Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional
Networks for Biomedical Image Segmentation. InMedical Image Computing and
Computer-Assisted Intervention – MICCAI 2015, Nassir Navab, Joachim Horneg-
ger, William M. Wells, and Alejandro F. Frangi (Eds.). Springer International
## Publishing, Cham, 234–241.
[54]Ruichen Sun, Kun Dong, and Jianfeng Zhao. 2023. DiffNILM: A Novel Framework
for Non-Intrusive Load Monitoring Based on the Conditional Diffusion Model.
Sensors23, 7 (2023).  doi:10.3390/s23073540
[55]  Stavros Sykiotis, Maria Kaselimi, Anastasios Doulamis, and Nikolaos Doulamis.
- ELECTRIcity: An Efficient Transformer for Non-Intrusive Load Monitoring.
Sensors22, 8 (2022).  doi:10.3390/s22082926
## [56]
Chang Wei Tan, Christoph Bergmeir, François Petitjean, and Geoffrey I. Webb.
- Time series extrinsic regression.Data Mining and Knowledge Discovery35,
3 (5 2021), 1032–1060.  doi:10.1007/s10618-021-00745-9
[57]L.N.  Sastry  Varanasi  and  Sri  Phani  Krishna  Karri.  2024.    STNILM:  Switch
Transformer based Non-Intrusive Load Monitoring for short and long dura-
tion  appliances.Sustainable Energy, Grids and Networks37  (2024),  101246.
doi:10.1016/j.segan.2023.101246
[58]Ashish  Vaswani,  Noam  Shazeer,  Niki  Parmar,  Jakob  Uszkoreit,  Llion  Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.  Attention Is All
You Need.CoRRabs/1706.03762 (2017). arXiv:1706.03762  http://arxiv.org/abs/
## 1706.03762
[59]Lingxiao Wang, Shiwen Mao, and R. Mark Nelms. 2022. Transformer for Nonin-
trusive Load Monitoring: Complexity Reduction and Transferability.IEEE Internet
of Things Journal9, 19 (2022), 18987–18997. doi:10.1109/JIOT.2022.3163347
## [60]
Min Xia, Wan’an Liu, Yiqing Xu, Ke Wang, and Xu Zhang. 2019. Dilated residual
attention network for load disaggregation.Neural Computing and Applications
31, 12 (12 2019), 8931–8953.  doi:10.1007/s00521-019-04414-3
[61]Zhenrui Yue, Camilo Requena Witzig, Daniel Jorde, and Hans-Arno Jacobsen.
- BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load
Monitoring. InProceedings of the 5th International Workshop on Non-Intrusive
Load Monitoring(Virtual Event, Japan)(NILM’20). Association for Computing
Machinery, New York, NY, USA, 89–93.  doi:10.1145/3427771.3429390
[62]George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
and Carsten Eickhoff. 2021. A Transformer-based Framework for Multivariate
Time Series Representation Learning. InProceedings of the 27th ACM SIGKDD
Conference on Knowledge Discovery & Data Mining(Virtual Event, Singapore)
(KDD ’21). Association for Computing Machinery, New York, NY, USA, 2114–2124.
doi:10.1145/3447548.3467401
[63]Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles
Sutton. 2018. Sequence-to-point learning with neural networks for non-intrusive
load monitoring. InProceedings of the Thirty-Second AAAI Conference on Arti-
ficial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence
Conference and Eighth AAAI Symposium on Educational Advances in Artificial
Intelligence(New Orleans, Louisiana, USA)(AAAI’18/IAAI’18/EAAI’18). AAAI
Press, Article 318, 8 pages.
## [64]
Bochao Zhao, Minxiang Ye, Lina Stankovic, and Vladimir Stankovic. 2020. Non-
intrusive load disaggregation solutions for very low-rate smart meter data.Ap-
plied Energy268 (2020), 114949.  doi:10.1016/j.apenergy.2020.114949
[65]Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong,
and Wan Zhang. 2020. Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting. InAAAI Conference on Artificial Intelligence.   https:
//api.semanticscholar.org/CorpusID:229156802
## [66]
## Ziwei Zhu, Mengran Zhou, Feng Hu, Kun Wang, Guangyao Zhou, Weile Kong,
Yijie Hu, and Enhan Cui. 2024.   TSILNet: A novel hybrid model for energy
disaggregation based on two-stage improved TCN combined with IECA-LSTM.
Building Simulation17, 11 (2024), 2083–2095.  doi:10.1007/s12273-024-1175-9

Conference acronym ’XX, June 03–05, 2018, Woodstock, NYA. Petralia et al.
Table 4: Number of learnable parameters (in millions) accord-
ing to input sequence window length.
## Models
Input subsequences푤length
## 푤=128푤=256푤=512
NILMFormer0.3850.3850.385
## BERT4NILM1.8931.911.943
## STNILM11.35911.37511.408
BiGRU0.4230.4230.423
## Energformer
## 0.5670.5670.567
## FCN0.1690.30.562
UNet-NILM2.4142.6763.202
TSILNet17.40234.24567.931
BiLSTM4.5178.72817.15
DiffNILM
## 9.2219.2219.221
DAResNet0.3310.4620.724
Table 5: Details of training hyperparameters
## Hyperparameters
Init. learning rate1e-4
## Scheduler
ReduceLROnPlateau
Patience reduce lr5
Batch size64
Max. number of epochs
## 50
Early stopping epochs10
Overall  metrics score# of parameters (in millions)
## 64961282566496128256
(b) Number of trainable
parameters (in millions)
## 풅
## 풎풐풅풆풍
## 풅
## 풎풐풅풆풍
# of Transformer layer(s)
(a) Overall metrics score (avg. of the
normalized MAE  and MR for 풘 =
ퟐퟓퟔ, over all the datasets and cases)
Figure 8: Influence of the number of Transformer layer(s)
according to the inner model dimension (푑
## 푚표푑푒푙
) in NILM-
Former on (a) the overall metrics disaggregation score; (b)
the number of learnable parameters.
Table 3: NILMFormer hyperparameters
## Hyperparameters
Embedding block
♯ResBlock4
## ♯filters72
kernel size3
Dilation rate{1,2,3,4}
Transformer block
♯Transformer Layers3
d_model96
## ♯heads8
PFFN ratio4
PFFN activationGeLU
## Dropout0.2
## Head
## ♯filters128
kernel size3
A  NILMFormer Architecture Details
This section provides additional details and insights about the NILM-
Former architecture.
## A.1  Hyperparameters
We report in Table 3 the list of hyperparameters used in NILM-
Former for our experiments.
## A.2  Hyperparameter Impact
We experimentally evaluate the impact of two main NILMFormer
hyperparameters (the inner dimension푑and the number of Trans-
former layers) on the disaggregation performances and the com-
plexity (in terms of the number of trainable parameters). More
specifically, we trained and evaluated NILMFormer in the setting
described in Section 5 (for all the datasets, cases, and with푤=256)
for the following inner dimension:푑={64,96,128,256}, and the
following number of Transformer layers푛
## 푙
## ={1,3,5,7}.
We reported the results using the overall metrics score, com-
puted by averaging the 2 metrics (MAE and MR, averaged to the
same range) and averaging the scores across the datasets and cases.
The heatmap in Figure 8 (a) indicates that combining 3 layers with
an inner dimension of푑=256yields the best disaggregation per-
formance. However, as reported in Figure 8 (b), this combination
induces a higher number of trainable parameters (3.55 million).
Therefore, due to the real-world deployment of our solution, we
opted for a more efficient configuration: 3 layers with an inner
dimension푑=96. This combination offers a good balance between
accuracy and a reduced number of parameters.
## B  Models Complexity
We examine the number of trainable parameters to compare the
complexity of the different NILM baselines used in our experiments.
Since this number is influenced by the subsequence window length
for certain baselines, we present the number of parameters accord-
ing to the window length (with푤={128,256,512}. The results,
reported in Table 4, indicate that FCN and BiGRU are the smallest
models in terms of trainable parameters. However, despite the use
of the Transformer architecture, NILMFormer’s number of trainable
parameters is kept small compared to the other baselines. Specifi-
cally, the second and third-best baselines, BERT4NILM and STNILM,
contain over 1 million and 11 million parameters, respectively.
## C  Reproducibility
All reported scores are averaged over three household-disjoint
train/test splits generated with seeds{0,1,2}.
[Deep-learning Baselines]All DL baselines were re-implemented
in PyTorch 2.5.1, exceptBERT4NILM[61], for which we used the
authors’ code. Training hyper-parameters appear in Table 5. Models
were optimised with Adam, a decaying learning-rate scheduler, and
early stopping.
[Other Baselines]ROCKET leverages the Aeon implementation [40],
while XGBoost uses the official Python package [10].