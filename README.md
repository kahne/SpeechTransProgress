End-to-End Speech Translation Progress
======

## Tutorial
* EACL 2021 tutorial: [Speech Translation](https://st-tutorial.github.io)
* Blog: [Getting Started with End-to-End Speech Translation](https://towardsdatascience.com/getting-started-with-end-to-end-speech-translation-3634c35a6561)
* ACL 2020 Theme paper: [Speech Translation and the End-to-End Promise: Taking Stock of Where We Are](https://arxiv.org/pdf/2004.06358.pdf)
* INTERSPEECH 2019 survey talk: [Spoken Language Translation](https://www.youtube.com/watch?v=beB5L6rsb0I)

## Data

| Corpus                                                                                                        |                                                                             Direction                                                                             |    Target     | Duration |                                                                                                            License |
|---------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:--------:|-------------------------------------------------------------------------------------------------------------------:|
| [CoVoST 2](https://arxiv.org/pdf/2007.10310.pdf)                                                              | {Fr, De, Es, Ca, It, Ru, Zh, Pt, Fa, Et, Mn, Nl, Tr, Ar, Sv, Lv, Sl, Ta, Ja, Id, Cy} -> En and En -> {De, Ca, Zh, Fa, Et, Mn, Tr, Ar, Sv, Lv, Sl, Ta, Ja, Id, Cy} |     Text      |  2880h   |                                                                                                                CC0 |
| [CVSS](https://arxiv.org/pdf/2201.03713.pdf) |                                    {Fr, De, Es, Ca, It, Ru, Zh, Pt, Fa, Et, Mn, Nl, Tr, Ar, Sv, Lv, Sl, Ta, Ja, Id, Cy} -> En                                     | Text & Speech |  1900h   |                                                                                                          CC BY 4.0 |                                                                                                                                                                   
| [mTEDx](https://arxiv.org/pdf/2102.01757.pdf)                                                                 |                                         {Es, Fr, Pt, It, Ru, El} -> En, {Fr, Pt, It} -> Es, Es -> {Fr, It}, {Es,Fr} -> Pt                                         |     Text      |   765h   |                                                                                                    CC BY-NC-ND 4.0 |
| [CoVoST](https://arxiv.org/pdf/2002.01320.pdf)                                                                |                                                        {Fr, De, Nl, Ru, Es, It, Tr, Fa, Sv, Mn, Zh} -> En                                                         |     Text      |   700h   |                                                                                                                CC0 |
| [MUST-C](https://www.aclweb.org/anthology/N19-1202.pdf) & [MUST-Cinema](https://arxiv.org/pdf/2002.10829.pdf) |                                                  En -> {De, Es, Fr, It, Nl, Pt, Ro, Ru, Ar, Cs, Fa, Tr, Vi, Zh}                                                   |     Text      |   504h   |                                                                                                    CC BY-NC-ND 4.0 |
| [How2](https://arxiv.org/pdf/1811.00347.pdf)                                                                  |                                                                             En -> Pt                                                                              |     Text      |   300h   |                                                                                             Youtube & CC BY-SA 4.0 |
| [Augmented LibriSpeech](https://arxiv.org/pdf/1802.03142.pdf)                                                 |                                                                             En -> Fr                                                                              |     Text      |   236h   |                                                                                                          CC BY 4.0 |
| [Europarl-ST](https://arxiv.org/pdf/1911.03167.pdf)                                                           |                                           {En, Fr, De, Es, It, Pt, Pl, Ro, Nl} -> {En, Fr, De, Es, It, Pt, Pl, Ro, Nl}                                            |     Text      |   280h   |                                                                                                       CC BY-NC 4.0 |
| [Kosp2e](https://arxiv.org/pdf/2107.02875.pdf)                                                                |                                                                             Ko -> En                                                                              |     Text      |   198h   |                                                                                                           Mixed CC |
| [Fisher + Callhome](https://www.seas.upenn.edu/~ccb/publications/improved-speech-to-speech-translation.pdf)   |                                                                             Es -> En                                                                              |     Text      | 160h+20h |                                                                                                                LDC |
| [MaSS](https://arxiv.org/pdf/1907.12895.pdf)                                                                  |                                                         parallel among En, Es, Eu, Fi, Fr, Hu, Ro and Ru                                                          | Text & Speech |   172h   |                                                                                                           Bible.is |
| [LibriVoxDeEn](https://arxiv.org/pdf/1910.07924.pdf)                                                          |                                                                             De -> En                                                                              |     Text      |   110h   |                                                                                                    CC BY-NC-SA 4.0 |
| [Prabhupadavani](https://arxiv.org/pdf/2201.11391.pdf) |                     parallel among En, Fr, De, Gu, Hi, Hu, Id, It, Lv, Lt, Ne, Fa, Pl, Pt, Ru, Sl, Sk, Es, Se, Ta, Te, Tr, Bg, Hr, Da and Nl                      |     Text |   94h    |  |
| [BSTC](https://arxiv.org/pdf/2104.03575.pdf)                                                                  |                                                                             Zh -> En                                                                              |     Text      |   68h    |                                                                                                                    |
| [LibriS2S](https://arxiv.org/pdf/2204.10593.pdf)                                                          |                                                                             De <-> En                                                                              |     Text & Speech      |   52h/57h   |                                                                                                    CC BY-NC-SA 4.0 |

## Toolkit
* [ESPNet-ST](https://github.com/espnet/espnet)
* [Fairseq S2T](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text)
* [NeurST](https://github.com/bytedance/neurst)

## Paper

### 2023
- [arXiv] [Tuning Large language model for End-to-end Speech Translation](https://arxiv.org/pdf/2310.02050.pdf)
- [arXiv] [Improving Speech Translation by Cross-Modal Multi-Grained Contrastive Learning](https://arxiv.org/pdf/2304.10309.pdf)
- [ICCV] [MixSpeech: Cross-Modality Self-Learning with Audio-Visual Stream Mixup for Visual Speech Translation and Recognition](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_MixSpeech_Cross-Modality_Self-Learning_with_Audio-Visual_Stream_Mixup_for_Visual_Speech_ICCV_2023_paper.pdf)
- [INTERSPEECH] [MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/anwar23_interspeech.pdf)
- [INTERSPEECH] [Modular Speech-to-Text Translation for Zero-Shot Cross-Modal Transfer](https://www.isca-speech.org/archive/pdfs/interspeech_2023/duquenne23_interspeech.pdf)
- [INTERSPEECH] [Joint Speech Translation and Named Entity Recognition](https://www.isca-speech.org/archive/pdfs/interspeech_2023/gaido23_interspeech.pdf)
- [INTERSPEECH] [StyleS2ST: Zero-shot Style Transfer for Direct Speech-to-speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/song23_interspeech.pdf)
- [INTERSPEECH] [Knowledge Distillation on Joint Task End-to-End Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/nayem23_interspeech.pdf)
- [INTERSPEECH] [GigaST: A 10,000-hour Pseudo Speech Translation Corpus](https://www.isca-speech.org/archive/pdfs/interspeech_2023/ye23b_interspeech.pdf)
- [INTERSPEECH] [Inter-connection: Effective Connection between Pre-trained Encoder and Decoder for Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/nishikawa23_interspeech.pdf)
- [INTERSPEECH] [AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/papi23_interspeech.pdf)
- [INTERSPEECH] [Incremental Blockwise Beam Search for Simultaneous Speech Translation with Controllable Quality-Latency Tradeoff](https://www.isca-speech.org/archive/pdfs/interspeech_2023/polak23_interspeech.pdf)
- [INTERSPEECH] [HK-LegiCoST: Leveraging Non-Verbatim Transcripts for Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2023/xiao23d_interspeech.pdf)
- [INTERSPEECH] [Learning When to Speak: Latency and Quality Trade-offs for Simultaneous Speech-to-Speech Translation with Offline Models](https://www.isca-speech.org/archive/pdfs/interspeech_2023/dugan23_interspeech.pdf)
- [ICML] [Pre-training for Speech Translation: CTC Meets Optimal Transport](http://proceedings.mlr.press/v202/le23a/le23a.pdf)
- [ACL] [UnitY: Two-pass Direct Speech-to-speech Translation with Discrete Units](https://aclanthology.org/2023.acl-long.872.pdf)
- [ACL] [Simple and effective unsupervised speech translation](https://aclanthology.org/2023.acl-long.602.pdf)
- [ACL] [BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric](https://aclanthology.org/2023.acl-long.504.pdf)
- [ACL] [SpeechMatrix: A Large-Scale Mined Corpus of Multilingual Speech-to-Speech Translations](https://aclanthology.org/2023.acl-long.899.pdf)
- [ACL] [Understanding and Bridging the Modality Gap for Speech Translation](https://aclanthology.org/2023.acl-long.884.pdf)
- [ACL] [Back Translation for Speech-to-text Translation Without Transcripts](https://aclanthology.org/2023.acl-long.251.pdf)
- [ACL] [AV-TranSpeech: Audio-Visual Robust Speech-to-Speech Translation](https://aclanthology.org/2023.acl-long.479.pdf)
- [ACL] [WACO: Word-Aligned Contrastive Learning for Speech Translation](https://arxiv.org/pdf/2212.09359.pdf)
- [ACL Findings] [Speech-to-Speech Translation for a Real-world Unwritten Language](https://aclanthology.org/2023.findings-acl.307.pdf)
- [ACL Findings] [CKDST: Comprehensively and Effectively Distill Knowledge from Machine Translation to End-to-End Speech Translation](https://aclanthology.org/2023.findings-acl.195.pdf)
- [ACL Findings] [Duplex Diffusion Models Improve Speech-to-Speech Translation](https://aclanthology.org/2023.findings-acl.509.pdf)
- [ACL Findings] [DUB: Discrete Unit Back-translation for Speech Translation](https://aclanthology.org/2023.findings-acl.447.pdf)
- [ACL Findings] [End-to-End Simultaneous Speech Translation with Differentiable Segmentation](https://aclanthology.org/2023.findings-acl.485.pdf)
- [ACL Findings] [Implicit Memory Transformer for Computationally Efficient Simultaneous Speech Translation](https://aclanthology.org/2023.findings-acl.816.pdf)
- [ICASSP] [Textless Direct Speech-to-Speech Translation with Discrete Speech Representation](https://ieeexplore.ieee.org/abstract/document/10096797)
- [ICASSP] [M3ST: Mix at Three Levels for Speech Translation](https://ieeexplore.ieee.org/abstract/document/10095090)

### 2022
- [arXiv] [Generating Synthetic Speech from SpokenVocab for Speech Translation](https://arxiv.org/pdf/2210.08174.pdf)
- [arXiv] [Joint Speech Transcription and Translation: Pseudo-Labeling with Out-of-Distribution Data](https://arxiv.org/pdf/2212.09982.pdf)
- [arXiv] [AdaTranS: Adapting with Boundary-based Shrinking for End-to-End Speech Translation](https://arxiv.org/pdf/2212.08911.pdf)
- [arXiv] [Attention as a guide for Simultaneous Speech Translation](https://arxiv.org/pdf/2212.07850.pdf)
- [arXiv] [BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric](https://arxiv.org/pdf/2212.08486.pdf)
- [arXiv] [Direct Speech-to-speech Translation without Textual Annotation using Bottleneck Features](https://arxiv.org/pdf/2212.05805.pdf)
- [arXiv] [M3ST: Mix at Three Levels for Speech Translation](https://arxiv.org/pdf/2212.03657.pdf)
- [arXiv] [Improving End-to-end Speech Translation by Leveraging Auxiliary Speech and Text Data](https://arxiv.org/pdf/2212.01778.pdf)
- [arXiv] [ArzEn-ST: A Three-way Speech Translation Corpus for Code-Switched Egyptian Arabic - English](https://arxiv.org/pdf/2211.12000.pdf)
- [arXiv] [M-Adapter: Modality Adaptation for End-to-End Speech-to-Text Translation](https://arxiv.org/pdf/2207.00952.pdf)
- [arXiv] [LibriS2S: A German-English Speech-to-Speech Translation Corpus](https://arxiv.org/pdf/2204.10593.pdf)
- [arXiv] [Blockwise Streaming Transformer for Spoken Language Understanding and Simultaneous Speech Translation](https://arxiv.org/pdf/2204.08920.pdf)
- [arXiv] [Does Simultaneous Speech Translation need Simultaneous Models?](https://arxiv.org/pdf/2204.03783.pdf)
- [arXiv] [Combining Spectral and Self-Supervised Features for Low Resource Speech Recognition and Translation](https://arxiv.org/pdf/2204.02470.pdf)
- [arXiv] [Large-Scale Streaming End-to-End Speech Translation with Neural Transducers](https://arxiv.org/pdf/2204.05352.pdf)
- [arXiv] [GigaST: A 10,000-hour Pseudo Speech Translation Corpus](https://arxiv.org/pdf/2204.03939.pdf)
- [arXiv] [Multilingual Simultaneous Speech Translation](https://arxiv.org/pdf/2203.14835.pdf)
- [arXiv] [Speech Segmentation Optimization using Segmented Bilingual Speech Corpus for End-to-end Speech Translation](https://arxiv.org/pdf/2203.15479.pdf)
- [arXiv] [Leveraging unsupervised and weakly-supervised data to improve direct speech-to-speech translation](https://arxiv.org/pdf/2203.13339.pdf)
- [arXiv] [Prabhupadavani: A Code-mixed Speech Translation Data for 25 Languages](https://arxiv.org/pdf/2201.11391.pdf)
- [arXiv] [CVSS Corpus and Massively Multilingual Speech-to-Speech Translation](https://arxiv.org/pdf/2201.03713.pdf)
- [arXiv] [SHAS: Approaching optimal Segmentation for End-to-End Speech Translation](https://arxiv.org/pdf/2202.04774.pdf)
- [EMNLP Findings] [RedApt: An Adaptor for WAV2VEC 2 Encoding Faster and Smaller Speech Translation without Quality Compromise](https://arxiv.org/pdf/2210.08475.pdf)
- [INTERSPEECH] [Exploring Continuous Integrate-and-Fire for Adaptive Simultaneous Speech Translation](https://arxiv.org/pdf/2204.09595.pdf)
- [NAACL] [Textless Speech-to-Speech Translation on Real Data](https://arxiv.org/pdf/2112.08352.pdf)
- [ICML] [Revisiting End-to-End Speech-to-Text Translation From Scratch](https://arxiv.org/pdf/2206.04571.pdf)
- [ICML] [Translatotron 2: Robust direct speech-to-speech translation](https://arxiv.org/pdf/2107.08661.pdf)
- [ACL] [Learning When to Translate for Streaming Speech](https://aclanthology.org/2022.acl-long.50.pdf)
- [ACL] [Sample, Translate, Recombine: Leveraging Audio Alignments for Data Augmentation in End-to-end Speech Translation](https://arxiv.org/abs/2203.08757)
- [ACL] [UniST: Unified End-to-end Model for Streaming and Non-streaming Speech Translation](https://arxiv.org/pdf/2109.07368.pdf)
- [ACL] [Direct speech-to-speech translation with discrete units](https://arxiv.org/pdf/2107.05604.pdf)
- [ACL] [STEMM: Self-learning with Speech-text Manifold Mixup for Speech Translation](https://arxiv.org/pdf/2203.10426.pdf)
- [ACL Findings] [End-to-End Speech Translation for Code Switched Speech](https://arxiv.org/pdf/2204.05076.pdf)
- [ICASSP] [Tackling data scarcity in speech translation using zero-shot multilingual machine translation techniques](https://arxiv.org/pdf/2201.11172.pdf)
- [NN] [Improving data augmentation for low resource speech-to-text translation with diverse paraphrasing](https://www.sciencedirect.com/science/article/pii/S0893608022000260)
- [AAAI] [Regularizing End-to-End Speech Translation with Triangular Decomposition Agreement](https://arxiv.org/pdf/2112.10991.pdf)

### 2021
- [arXiv] [Efficient Transformer for Direct Speech Translation](https://arxiv.org/pdf/2107.03069.pdf)
- [arXiv] [Zero-shot Speech Translation](https://arxiv.org/pdf/2107.06010.pdf)
- [arXiv] [Direct Simultaneous Speech-to-Speech Translation with Variational Monotonic Multihead Attention](https://arxiv.org/pdf/2110.08250.pdf)
- [ASRU] [Fast-MD: Fast Multi-Decoder End-to-End Speech Translation with Non-Autoregressive Hidden Intermediates](https://arxiv.org/pdf/2109.12804.pdf)
- [ASRU] [Assessing Evaluation Metrics for Speech-to-Speech Translation](https://arxiv.org/pdf/2110.13877.pdf)
- [ASRU] [Enabling Zero-shot Multilingual Spoken Language Translation with Language-Specific Encoders and Decoders](https://arxiv.org/pdf/2011.01097.pdf)
- [ICNLSP] [Beyond Voice Activity Detection: Hybrid Audio Segmentation for Direct Speech Translation](https://arxiv.org/pdf/2104.11710.pdf)
- [INTERSPEECH] [Impact of Encoding and Segmentation Strategies on End-to-End Simultaneous Speech Translation](https://arxiv.org/pdf/2104.14470.pdf)
- [EMNLP] [Speechformer: Reducing Information Loss in Direct Speech Translation](https://arxiv.org/pdf/2109.04574.pdf)
- [EMNLP] [Is "moby dick" a Whale or a Bird? Named Entities and Terminology in Speech Translation](https://arxiv.org/pdf/2109.07439.pdf)
- [EMNLP] [Mutual-Learning Improves End-to-End Speech Translation](https://aclanthology.org/2021.emnlp-main.325.pdf)
- [INTERSPEECH] [End-to-end Speech Translation via Cross-modal Progressive Training](https://arxiv.org/pdf/2104.10380.pdf)
- [INTERSPEECH] [CoVoST 2 and Massively Multilingual Speech-to-Text Translation](https://arxiv.org/pdf/2007.10310.pdf)
- [INTERSPEECH] [The Multilingual TEDx Corpus for Speech Recognition and Translation](https://arxiv.org/pdf/2102.01757.pdf)
- [INTERSPEECH] [Large-Scale Self-and Semi-Supervised Learning for Speech Translation](https://arxiv.org/pdf/2104.06678.pdf)
- [INTERSPEECH] [Kosp2e: Korean Speech to English Translation Corpus](https://arxiv.org/pdf/2107.02875.pdf)
- [INTERSPEECH] [AlloST: Low-resource Speech Translation without Source Transcription](https://arxiv.org/pdf/2105.00171.pdf)
- [INTERSPEECH] [SpecRec: An Alternative Solution for Improving End-to-End Speech-to-Text Translation via Spectrogram Reconstruction](https://www.isca-speech.org/archive/pdfs/interspeech_2021/chen21i_interspeech.pdf)
- [INTERSPEECH] [Optimally Encoding Inductive Biases into the Transformer Improves End-to-End Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2021/vyas21_interspeech.pdf)
- [INTERSPEECH] [ASR Posterior-based Loss for Multi-task End-to-end Speech Translation](https://www.isca-speech.org/archive/pdfs/interspeech_2021/ko21_interspeech.pdf)
- [AMTA] [Simultaneous Speech Translation for Live Subtitling: from Delay to Display](https://arxiv.org/pdf/2107.08807.pdf)
- [ACL] [Stacked Acoustic-and-Textual Encoding: Integrating the Pre-trained Models into Speech Translation Encoders](https://arxiv.org/pdf/2105.05752.pdf)
- [ACL] [Multilingual Speech Translation with Efficient Finetuning of Pretrained Models](https://arxiv.org/pdf/2010.12829.pdf)
- [ACL] [Lightweight Adapter Tuning for Multilingual Speech Translation](https://arxiv.org/pdf/2106.01463.pdf)
- [ACL] [Cascade versus Direct Speech Translation: Do the Differences Still Make a Difference?](https://arxiv.org/pdf/2106.01045.pdf)
- [ACL] [Improving Speech Translation by Understanding and Learning from the Auxiliary Text Translation Task](https://aclanthology.org/2021.acl-long.328.pdf)
- [ACL] [Beyond Sentence-Level End-to-End Speech Translation: Context Helps](https://aclanthology.org/2021.acl-long.200.pdf)
- [ACL Findings] [Direct Simultaneous Speech-to-Text Translation Assisted by Synchronized Streaming ASR](https://arxiv.org/pdf/2106.06636.pdf)
- [ACL Findings] [AdaST: Dynamically Adapting Encoder States in the Decoder for End-to-End Speech-to-Text Translation]()
- [ACL Findings] [RealTranS: End-to-End Simultaneous Speech Translation with Convolutional Weighted-Shrinking Transformer](https://arxiv.org/pdf/2106.04833.pdf)
- [ACL Findings] [Learning Shared Semantic Space for Speech-to-Text Translation](https://arxiv.org/pdf/2105.03095.pdf)
- [ACL Findings] [Investigating the Reordering Capability in CTC-based Non-Autoregressive End-to-End Speech Translation](https://arxiv.org/pdf/2105.04840.pdf)
- [ACL Findings] [How to Split: the Effect of Word Segmentation on Gender Bias in Speech Translation](https://arxiv.org/pdf/2105.13782.pdf)
- [ACL Demo] [NeurST: Neural Speech Translation Toolkit](https://arxiv.org/pdf/2012.10018.pdf)
- [ICML] [Fused Acoustic and Text Encoding for Multimodal Bilingual Pretraining and Speech Translation](https://arxiv.org/pdf/2102.05766.pdf)
- [NAACL] [Source and Target Bidirectional Knowledge Distillation for End-to-end Speech Translation](https://arxiv.org/pdf/2104.06457.pdf)
- [NAACL] [Searchable Hidden Intermediates for End-to-End Models of Decomposable Sequence Tasks](https://arxiv.org/pdf/2105.00573.pdf)
- [NAACL AutoSimTrans] [BSTC: A Large-Scale Chinese-English Speech Translation Dataset](https://arxiv.org/pdf/2104.03575.pdf)
- [AmericasNLP] [Highland Puebla Nahuatl–Spanish Speech Translation Corpus for Endangered Language Documentation](https://www.aclweb.org/anthology/2021.americasnlp-1.7.pdf)
- [ICASSP] [Task Aware Multi-Task Learning for Speech to Text Tasks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9414703)
- [ICASSP] [A General Multi-Task Learning Framework to Leverage Text Data for Speech to Text Tasks](https://arxiv.org/pdf/2010.11338.pdf)
- [ICASSP] [An Empirical Study of End-to-end Simultaneous Speech Translation Decoding Strategies](https://arxiv.org/pdf/2103.03233.pdf)
- [ICASSP] [Streaming Simultaneous Speech Translation with Augmented Memory Transformer](https://arxiv.org/pdf/2011.00033.pdf)
- [ICASSP] [Orthros: Non-autoregressive End-to-end Speech Translation with Dual-decoder](https://arxiv.org/pdf/2010.13047.pdf)
- [ICASSP] [Cascaded Models With Cyclic Feedback For Direct Speech Translation](https://arxiv.org/pdf/2010.11153.pdf)
- [ICASSP] [Jointly Trained Transformers models for Spoken Language Translation](https://arxiv.org/pdf/2004.12111.pdf)
- [ICASSP] [Efficient Use of End-to-end Data in Spoken Language Processing](https://ieeexplore.ieee.org/document/9414510)
- [EACL] [CTC-based Compression for Direct Speech Translation](https://arxiv.org/pdf/2102.01578.pdf)
- [EACL] [Streaming Models for Joint Speech Recognition and Translation](https://arxiv.org/pdf/2101.09149.pdf)
- [IberSPEECH] [mintzai-ST: Corpus and Baselines for Basque-Spanish Speech Translation](https://www.isca-speech.org/archive/IberSPEECH_2021/pdfs/41.pdf)
- [AAAI] [Consecutive Decoding for Speech-to-text Translation](https://arxiv.org/pdf/2009.09737.pdf)
- [AAAI] [UWSpeech: Speech to Speech Translation for Unwritten Languages](https://arxiv.org/pdf/2006.07926.pdf)
- [AAAI] ["Listen, Understand and Translate": Triple Supervision Decouples End-to-end Speech-to-text Translation](https://arxiv.org/pdf/2009.09704.pdf)
- [SLT] [Tight Integrated End-to-End Training for Cascaded Speech Translation](https://arxiv.org/pdf/2011.12167.pdf)
- [SLT] [Transformer-based Direct Speech-to-speech Translation with Transcoder](https://ahcweb01.naist.jp/papers/conference/2021/202101_SLT_takatomo-k/202101_SLT_takatomo-k.paper.pdf)

### 2020
- [arXiv] [Bridging the Modality Gap for Speech-to-Text Translation](https://arxiv.org/pdf/2010.14920.pdf)
- [arXiv] [CSTNet: Contrastive Speech Translation Network for Self-Supervised Speech Representation Learning](https://arxiv.org/pdf/2006.02814.pdf)
- [CLiC-IT] [On Knowledge Distillation for Direct Speech Translation](https://arxiv.org/pdf/2012.04964.pdf)
- [COLING] [Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation](https://arxiv.org/pdf/2011.00747.pdf)
- [COLING] [Breeding Gender-aware Direct Speech Translation Systems](https://arxiv.org/pdf/2012.04955.pdf)
- [AACL] [SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation](https://www.aclweb.org/anthology/2020.aacl-main.58.pdf)
- [AACL Demo] [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/pdf/2010.05171.pdf)
- [EMNLP] [Effectively pretraining a speech translation decoder with Machine Translation data](https://www.aclweb.org/anthology/2020.emnlp-main.644.pdf)
- [EMNLP Findings] [Adaptive Feature Selection for End-to-End Speech Translation](https://arxiv.org/pdf/2010.08518.pdf)
- [AMTA] [On Target Segmentation for Direct Speech Translation](https://arxiv.org/pdf/2009.04707.pdf)
- [INTERSPEECH] [Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection](https://arxiv.org/pdf/2005.11185.pdf)
- [INTERSPEECH] [Relative Positional Encoding for Speech Recognition and Direct Translation](https://indico2.conference4me.psnc.pl/event/35/contributions/2722/attachments/329/354/Mon-1-1-7.pdf)
- [INTERSPEECH] [Contextualized Translation of Automatically Segmented Speech](https://arxiv.org/pdf/2008.02270.pdf)
- [INTERSPEECH] [Self-Training for End-to-End Speech Translation](https://arxiv.org/pdf/2006.02490.pdf)
- [INTERSPEECH] [Improving Cross-Lingual Transfer Learning for End-to-End Speech Recognition with Speech Translation](https://arxiv.org/pdf/2006.05474.pdf)
- [INTERSPEECH] [Self-Supervised Representations Improve End-to-End Speech Translation](https://arxiv.org/pdf/2006.12124.pdf)
- [INTERSPEECH] [Investigating Self-Supervised Pre-Training for End-to-End Speech Translation](http://www.interspeech2020.org/uploadfile/pdf/Tue-1-1-3.pdf)
- [TACL] [Consistent Transcription and Translation of Speech](https://arxiv.org/pdf/2007.12741.pdf)
- [ACL] [Worse WER, but Better BLEU? Leveraging Word Embedding as Intermediate in Multitask End-to-End Speech Translation](https://arxiv.org/pdf/2005.10678.pdf)
- [ACL] [Phone Features Improve Speech Translation](https://arxiv.org/pdf/2005.13681.pdf)
- [ACL] [Curriculum Pre-training for End-to-End Speech Translation](https://arxiv.org/pdf/2004.10093.pdf)
- [ACL] [SimulSpeech: End-to-End Simultaneous Speech to Text Translation](https://www.aclweb.org/anthology/2020.acl-main.350.pdf)
- [ACL] [Gender in Danger? Evaluating Speech Translation Technology on the MuST-SHE Corpus](https://aclanthology.org/2020.acl-main.619.pdf)
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
- [INTERSPEECH] [End-to-End Speech Translation with Knowledge Distillation](https://arxiv.org/pdf/1904.08075.pdf)
- [INTERSPEECH] [Adapting Transformer to End-to-end Spoken Language Translation](https://www.isca-speech.org/archive/INTERSPEECH_2019/pdfs/3045.pdf)
- [INTERSPEECH] [Direct speech-to-speech translation with a sequence-to-sequence model](https://arxiv.org/pdf/1904.06037.pdf)
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
- [INTERSPEECH] [Low-Resource Speech-to-Text Translation](https://arxiv.org/pdf/1803.09164.pdf)
- [LREC] [Augmenting Librispeech with French Translations: A Multimodal Corpus for Direct Speech Translation Evaluation](https://arxiv.org/pdf/1802.03142.pdf)
- [NAACL] [Tied multitask learning for neural speech translation](https://www.aclweb.org/anthology/N18-1008.pdf)
- [ICASSP] [End-to-End Automatic Speech Translation of Audiobooks](https://arxiv.org/pdf/1802.04200.pdf)

### 2017
- [INTERSPEECH] [Sequence-to-Sequence Models Can Directly Translate Foreign Speech](https://arxiv.org/pdf/1703.08581.pdf)
- [INTERSPEECH] [Structured-based Curriculum Learning for End-to-end English-Japanese Speech Translation](https://arxiv.org/abs/1802.06003)
- [EACL] [Towards speech-to-text translation without speech recognition](https://arxiv.org/pdf/1702.03856.pdf)

### 2016
- [NIPS Workshop] [Listen and translate: A proof of concept for end-to-end speech-to-text translation](https://arxiv.org/pdf/1612.01744.pdf)
- [NAACL] [An Attentional Model for Speech Translation Without Transcription](https://www.aclweb.org/anthology/N16-1109.pdf)

### 2013
- [IWSLT] [Improved Speech-to-Text Translation with the Fisher and Callhome Spanish–English Speech Translation Corpus](https://www.seas.upenn.edu/~ccb/publications/improved-speech-to-speech-translation.pdf)


# Contact

Changhan Wang ([wangchanghan@gmail.com](mailto:wangchanghan@gmail.com))
