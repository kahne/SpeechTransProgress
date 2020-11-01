End-to-End Speech-to-Text/Speech Translation Progress
======

## Tutorial
* Interspeech 2019 survey talk: [Spoken Language Translation](https://www.youtube.com/watch?v=beB5L6rsb0I)
* Blog: [Getting Started with End-to-End Speech Translation](https://towardsdatascience.com/getting-started-with-end-to-end-speech-translation-3634c35a6561)
* ACL 2020 Theme paper: [Speech Translation and the End-to-End Promise: Taking Stock of Where We Are](https://arxiv.org/pdf/2004.06358.pdf)

## Data

| Corpus | Direction | Target | Duration | License |
| ------ |:-------:|:-----:|:-----:|----:|
| [CoVoST 2](https://arxiv.org/pdf/2007.10310.pdf) | {Fr, De, Es, Ca, It, Ru, Zh, Pt, Fa, Et, Mn, Nl, Tr, Ar, Sv, Lv, Sl, Ta, Ja, Id, Cy} -> En and En -> {De, Ca, Zh, Fa, Et, Mn, Tr, Ar, Sv, Lv, Sl, Ta, Ja, Id, Cy} | Text | 2880h | CC0 |
| [CoVoST](https://arxiv.org/pdf/2002.01320.pdf) | {Fr, De, Nl, Ru, Es, It, Tr, Fa, Sv, Mn, Zh} -> En | Text | 700h | CC0 |
| [MUST-C](https://www.aclweb.org/anthology/N19-1202.pdf) & [MUST-Cinema](https://arxiv.org/pdf/2002.10829.pdf) | En -> {De, Es, Fr, It, Nl, Pt, Ro, Ru} | Text | 504h | CC BY-NC-ND 4.0 |
| [How2](https://arxiv.org/pdf/1811.00347.pdf) | En -> Pt | Text | 300h | Youtube & CC BY-SA 4.0 |
| [Augmented Librispeech](https://arxiv.org/pdf/1802.03142.pdf) | En -> Fr | Text | 236h | CC BY 4.0 |
| [Europarl-ST](https://arxiv.org/pdf/1911.03167.pdf) | {En, Fr, De, Es, It, Pt} -> {En, Fr, De, Es, It, Pt} | Text | 200h | CC BY-NC 4.0 |
| [Fisher + Callhome](https://www.seas.upenn.edu/~ccb/publications/improved-speech-to-speech-translation.pdf) | Es -> En | Text | 160h+20h | LDC |
| [MaSS](https://arxiv.org/pdf/1907.12895.pdf) | {En, Es, Eu, Fi, Fr, Hu, Ro, Ru} -> {En, Es, Eu, Fi, Fr, Hu, Ro, Ru} | Text & Speech | 172h | Bible.is |
| [LibriVoxDeEn](https://arxiv.org/pdf/1910.07924.pdf) | De -> En | Text | 110h | CC BY-NC-SA 4.0 |

## Paper

### 2020
- [arXiv] [Orthros: Non-autoregressive End-to-end Speech Translation with Dual-decoder](https://arxiv.org/pdf/2010.13047.pdf)
- [arXiv] [Bridging the Modality Gap for Speech-to-Text Translation](https://arxiv.org/pdf/2010.14920.pdf)
- [arXiv] [A General Multi-Task Learning Framework to Leverage Text Data for Speech to Text Tasks](https://arxiv.org/pdf/2010.11338.pdf)
- [arXiv] [Cascaded Models With Cyclic Feedback For Direct Speech Translation](https://arxiv.org/pdf/2010.11153.pdf)
- [arXiv] [MAM: Masked Acoustic Modeling for End-to-End Speech-to-Text Translation](https://arxiv.org/pdf/2010.11445.pdf)
- [arXiv] [TED: Triple Supervision Decouples End-to-end Speech-to-text Translation](https://arxiv.org/pdf/2009.09704.pdf)
- [arXiv] [On Target Segmentation for Direct Speech Translation](https://arxiv.org/pdf/2009.04707.pdf)
- [arXiv] [CoVoST 2 and Massively Multilingual Speech-to-Text Translation](https://arxiv.org/pdf/2007.10310.pdf)
- [arXiv] [CSTNet: Contrastive Speech Translation Network for Self-Supervised Speech Representation Learning](https://arxiv.org/pdf/2006.02814.pdf)
- [arXiv] [UWSpeech: Speech to Speech Translation for Unwritten Languages](https://arxiv.org/pdf/2006.07926.pdf)
- [arXiv] [Relative Positional Encoding for Speech Recognition and Direct Translation](https://arxiv.org/pdf/2005.09940.pdf)
- [arXiv] [Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection](https://arxiv.org/pdf/2005.11185.pdf)
- [EMNLP Findings] [Adaptive Feature Selection for End-to-End Speech Translation](https://arxiv.org/pdf/2010.08518.pdf)
- [AACL Demo] [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/pdf/2010.05171.pdf)
- [Interspeech] [Relative Positional Encoding for Speech Recognition and Direct Translation](https://indico2.conference4me.psnc.pl/event/35/contributions/2722/attachments/329/354/Mon-1-1-7.pdf)
- [Interspeech] [Contextualized Translation of Automatically Segmented Speech](https://arxiv.org/pdf/2008.02270.pdf)
- [Interspeech] [Self-Training for End-to-End Speech Translation](https://arxiv.org/pdf/2006.02490.pdf)
- [Interspeech] [Improving Cross-Lingual Transfer Learning for End-to-End Speech Recognition with Speech Translation](https://arxiv.org/pdf/2006.05474.pdf)
- [Interspeech] [Self-Supervised Representations Improve End-to-End Speech Translation](https://arxiv.org/pdf/2006.12124.pdf)
- [TACL] [Consistent Transcription and Translation of Speech](https://arxiv.org/pdf/2007.12741.pdf)
- [ACL] [Worse WER, but Better BLEU? Leveraging Word Embedding as Intermediate in Multitask End-to-End Speech Translation](https://arxiv.org/pdf/2005.10678.pdf)
- [ACL] [Phone Features Improve Speech Translation](https://arxiv.org/pdf/2005.13681.pdf)
- [ACL] [Curriculum Pre-training for End-to-End Speech Translation](https://arxiv.org/pdf/2004.10093.pdf)
- [ACL] [SimulSpeech: End-to-End Simultaneous Speech to Text Translation](https://www.aclweb.org/anthology/2020.acl-main.350.pdf)
- [ACL Theme] [Speech Translation and the End-to-End Promise: Taking Stock of Where We Are](https://arxiv.org/pdf/2004.06358.pdf)
- [ACL Demo] [ESPnet-ST: All-in-One Speech Translation Toolkit](https://arxiv.org/pdf/2004.10234.pdf)
- [LREC] [CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus](https://arxiv.org/pdf/2002.01320.pdf)
- [LREC] [MuST-Cinema: a Speech-to-Subtitles corpus](https://arxiv.org/pdf/2002.10829.pdf)
- [LREC] [MaSS: A Large and Clean Multilingual Corpus of Sentence-aligned Spoken Utterances Extracted from the Bible](https://arxiv.org/pdf/1907.12895.pdf)
- [LREC] [LibriVoxDeEn: A Corpus for German-to-English Speech Translation and Speech Recognition](https://arxiv.org/pdf/1910.07924.pdf)
- [ICASSP] [Europarl-ST: A Multilingual Corpus For Speech Translation Of Parliamentary Debates](https://arxiv.org/pdf/1911.03167.pdf)
- [ICASSP] [Instance-Based Model Adaptation For Direct Speech Translation](https://arxiv.org/pdf/1910.10663.pdf)
- [ICASSP] [Data Efficient Direct Speech-to-Text Translation with Modality Agnostic Meta-Learning](https://arxiv.org/pdf/1911.04283.pdf)
- [ICASSP] [Analyzing ASR pretraining for low-resource speech-to-text translation](https://arxiv.org/pdf/1910.10762.pdf)
- [ICASSP] [End-to-End Speech Translation with Self-Contained Vocabulary Manipulation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053431)
- [AAAI] [Bridging the Gap between Pre-Training and Fine-Tuning for End-to-End Speech Translation](https://arxiv.org/pdf/1909.07575.pdf)
- [AAAI] [Synchronous Speech Recognition and Speech-to-Text Translation with Interactive Decoding](https://arxiv.org/pdf/1912.07240.pdf)

### 2019
- [ASRU] [One-To-Many Multilingual End-to-end Speech Translation](https://arxiv.org/pdf/1910.03320.pdf)
- [ASRU] [Multilingual End-to-End Speech Translation](https://arxiv.org/pdf/1910.00254.pdf)
- [ASRU] [Speech-to-speech Translation between Untranscribed Unknown Languages](https://arxiv.org/pdf/1910.00795.pdf)
- [ASRU] [A Comparative Study on End-to-end Speech to Text Translation](https://arxiv.org/pdf/1911.08870.pdf)
- [IWSLT] [Harnessing Indirect Training Data for End-to-End Automatic Speech Translation: Tricks of the Trade](https://arxiv.org/pdf/1909.06515.pdf)
- [IWSLT] [On Using SpecAugment for End-to-End Speech Translation](https://arxiv.org/pdf/1911.08876.pdf)
- [Interspeech] [End-to-End Speech Translation with Knowledge Distillation](https://arxiv.org/pdf/1904.08075.pdf)
- [Interspeech] [Adapting Transformer to End-to-end Spoken Language Translation](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)
- [Interspeech] [Direct speech-to-speech translation with a sequence-to-sequence model](https://arxiv.org/pdf/1904.06037.pdf)
- [ACL] [Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation](https://www.aclweb.org/anthology/P19-1179.pdf)
- [ACL] [Attention-Passing Models for Robust and Data-Efficient End-to-End Speech Translation](https://arxiv.org/pdf/1904.07209.pdf)
- [NAACL] [Pre-training on High-Resource Speech Recognition Improves Low-Resource Speech-to-Text Translation](https://www.aclweb.org/anthology/N19-1006.pdf)
- [NAACL] [MuST-C: a Multilingual Speech Translation Corpus](https://www.aclweb.org/anthology/N19-1202.pdf)
- [NAACL] [Fluent Translations from Disfluent Speech in End-to-End Speech Translation](https://arxiv.org/pdf/1906.00556.pdf)
- [ICASSP] [Leveraging Weakly Supervised Data to Improve End-to-End Speech-to-Text Translation](https://arxiv.org/pdf/1811.02050.pdf)
- [ICASSP] [Towards unsupervised speech-to-text translation](https://arxiv.org/pdf/1811.01307.pdf)
- [ICASSP] [Towards End-to-end Speech-to-text Translation with Two-pass Decoding](https://orbxball.github.io/pub/icassp-2019.pdf)

### 2018
- [NIPS] [How2: A Large-scale Dataset for Multimodal Language Understanding](https://arxiv.org/pdf/1811.00347.pdf)
- [IberSPEECH] [End-to-End Speech Translation with the Transformer](https://pdfs.semanticscholar.org/5253/3e5ffc2f9b635fa21259f9749609b1f9dfa1.pdf?_ga=2.259261770.172395283.1580036833-1842396350.1580036833)
- [Interspeech] [Low-Resource Speech-to-Text Translation](https://arxiv.org/pdf/1803.09164.pdf)
- [LREC] [Augmenting Librispeech with French Translations: A Multimodal Corpus for Direct Speech Translation Evaluation](https://arxiv.org/pdf/1802.03142.pdf)
- [NAACL] [Tied multitask learning for neural speech translation](https://www.aclweb.org/anthology/N18-1008.pdf)
- [ICASSP] [End-to-End Automatic Speech Translation of Audiobooks](https://arxiv.org/pdf/1802.04200.pdf)

### 2017
- [Interspeech] [Sequence-to-Sequence Models Can Directly Translate Foreign Speech](https://arxiv.org/pdf/1703.08581.pdf)
- [Interspeech] [Structured-based Curriculum Learning for End-to-end English-Japanese Speech Translation](https://arxiv.org/abs/1802.06003)
- [EACL] [Towards speech-to-text translation without speech recognition](https://arxiv.org/pdf/1702.03856.pdf)

### 2016
- [NIPS Workshop] [Listen and translate: A proof of concept for end-to-end speech-to-text translation](https://arxiv.org/pdf/1612.01744.pdf)
- [NAACL] [An Attentional Model for Speech Translation Without Transcription](https://www.aclweb.org/anthology/N16-1109.pdf)

### 2013
- [IWSLT] [Improved Speech-to-Text Translation with the Fisher and Callhome Spanish–English Speech Translation Corpus](https://www.seas.upenn.edu/~ccb/publications/improved-speech-to-speech-translation.pdf)


# Contact

Changhan Wang ([wangchanghan@gmail.com](mailto:wangchanghan@gmail.com))
