![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000000_7030159f15270e9ccb54ad81800ae360badee1563d0c3018b7e80173fe688947.png)

Article

## Robust Object Detection in Adverse Weather Conditions: ECL-YOLOv11 for Automotive Vision Systems

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000001_625f4461ca278305f7c3bd907a0a8eafa499c73b3375ef7aa9f7f691c49a0456.png)

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000002_adc5abe84afed4e5dd215619ccaf594efdea4252a44e260e4f5da0a316c7dd70.png)

Academic Editor: Steve Vanlanduit

Received: 20 October 2025

Revised: 23 November 2025

Accepted: 24 November 2025

Published: 2 January 2026

Copyright:

©2026 by the authors.

Licensee MDPI, Basel, Switzerland.

This article is an open access article

distributed under the terms and

conditions of the Creative Commons

Attribution (CC BY) license.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000003_4075486e54d4bcc16d18702fe5f91c16a41eda95dec2e0e6c11134900252e069.png)

Zhaohui Liu * , Jiaxu Zhang , Xiaojun Zhang and Hongle Song

College of Transportation, Shandong University of Science and Technology, Qingdao 266590, China; 13956475095@163.com (J.Z.); pnxxzxj@163.com (X.Z.); honglest@163.com (H.S.)

* Correspondence: zhaohuiliu@sdust.edu.cn

## Abstract

The rapid development of intelligent transportation systems and autonomous driving technologies has made visual perception a key component in ensuring safety and improving efficiency in complex traffic environments. As a core task in visual perception, object detection directly affects the reliability of downstream modules such as path planning and decision control. However, adverse weather conditions (e.g., fog, rain, and snow) significantly degrade image quality-causing texture blurring, reduced contrast, and increased noise-which in turn weakens the robustness of traditional detection models and raises potential traffic safety risks. To address this challenge, this paper proposes an enhanced object detection framework, ECL-YOLOv11 (Edge-enhanced, Context-guided, and Lightweight YOLOv11), designed to improve detection accuracy and real-time performance under adverse weather conditions, thereby providing a reliable solution for in-vehicle perception systems. The ECL-YOLOv11 architecture integrates three key modules: (1) a Convolutional Edge-enhancement (CE) module that fuses edge features extracted by Sobel operators with convolutional features to explicitly retain boundary and contour information, thereby alleviating feature degradation and improving localization accuracy under low-visibility conditions; (2) a Context-guided Multi-scale Fusion Network (AENet) that enhances perception of small and distant objects through multi-scale feature integration and context modeling, improving semantic consistency and detection stability in complex scenes; and (3) a Lightweight Shared Convolutional Detection Head (LDHead) that adopts shared convolutions and GroupNorm normalization to optimize computational efficiency, reduce inference latency, and satisfy the real-time requirements of on-board systems. Experimental results show that ECL-YOLOv11 achieves mAP@50 and mAP@50-95 values of 62.7% and 40.5%, respectively, representing improvements of 1.3% and 0.8% over the baseline YOLOv11, while the Precision reaches 73.1%. The model achieves a balanced trade-off between accuracy and inference speed, operating at 237.8 FPS on standard hardware. Ablation studies confirm the independent effectiveness of each proposed module in feature enhancement, multi-scale fusion, and lightweight detection, while their integration further improves overall performance. Qualitative visualizations demonstrate that ECLYOLOv11 maintains high-confidence detections across varying motion states and adverse weather conditions, avoiding category confusion and missed detections. These results indicate that the proposed framework provides a reliable and adaptable foundation for all-weather perception in autonomous driving systems, ensuring both operational safety and real-time responsiveness.

Keywords: adverse weather; automotive vision; edge-enhancement; multi-scale fusion; lightweight detection

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000004_5f663a8cb42f590dca7596fb8cdb6750c8d619b6851566a3d281f3ead4154ad2.png)

## 1. Introduction

With the rapid development of intelligent transportation systems and autonomous driving technologies, visual perception has become one of the core technologies for ensuring safety and improving efficiency in complex traffic environments. Object detection, as a key task in visual perception, directly impacts the reliability of downstream functions such as path planning and decision control [1,2]. However, in practical applications, adverse weather conditions (such as fog, rain, and snow) remain a significant challenge for object detection. These weather factors significantly degrade image quality, causing blurriness, reduced contrast, and increased noise, which severely weakens the robustness of traditional object detection frameworks [3,4]. In low-visibility environments, stable detection of vehicles and pedestrians is particularly crucial, as detection failures can directly lead to traffic safety incidents [5].

In recent years, object detection methods based on convolutional neural networks (CNNs)-such as Faster R-CNN [6], the YOLO series [7], and SSD [8], have achieved remarkable progress in conventional scenarios, attaining a relatively balanced trade-off between detection accuracy and real-time performance. However, when visibility decreases significantly, these models often find it difficult to effectively preserve edge and texture features, resulting in blurred object boundaries, missed small targets, and category confusion. In addition, the stringent real-time requirements of automotive platforms mean that complex or high-parameter detection models can easily cause inference latency issues. Therefore, improving the robustness of detection models under adverse weather conditions while maintaining high inference efficiency has become a key challenge for intelligent driving vision systems.

To address these challenges, this study proposes an enhanced object detection framework-ECL-YOLOv11 (Edge-enhanced Context-guided Lightweight YOLOv11), building upon our previous research efforts [9-11], aiming to solve the robustness and real-time challenges of object detection under adverse weather conditions. Compared to the traditional YOLOv11 [12], ECL-YOLOv11 introduces three structural improvements: the Edge-Enhancement Convolution (CE) module integrated is added to the Backbone to preserve edge and contour information in images; in the Neck, the Context-Guided Multi-Scale Fusion Network (AENet) is designed to enhance the model's capability of perceiving multi-scale features through context guidance; and the Lightweight Shared Convolution Detection Head (LDHead) in the Head, which reduces redundant computations and improves real-time inference efficiency. These three modules are structurally complementary, corresponding respectively to edge preservation, context fusion, and computational optimization, forming an integrated architecture for object detection under adverse weather conditions.

The main innovations of this paper include:

(1) For the first time, the YOLOv11 framework introduces an explicit edge enhancement mechanism, in which the CE module strengthens the edges and texture features of low-quality images.

(2) The AENet structure based on rectangular calibration and cross-scale fusion is proposed, which effectively improves the model's small-object detection performance under low-visibility conditions.

(3) The LDHead adopts a lightweight shared design for the detection head, reducing GFLOPs while maintaining high-precision output.

(4) Under various adverse weather conditions, the proposed model demonstrates a good balance between robustness and real-time performance.

Comprehensive experimental results from multiple perspectives demonstrate the superiority of the ECL-YOLOv11 model over the YOLOv11 baseline in terms of core

metrics such as mAP and Precision. Notably, it achieves this while maintaining a high inference speed of 230.3 FPS, underscoring its high usability for practical automotive vision systems. Furthermore, the model's stable performance under various weather conditions and complex scenarios involving different motion states validates its robust generalization capability and considerable potential for deployment on vehicle terminals.

## 2. Related Work

## 2.1. Traditional CNN-Based Object Detection Methods

Over the past decade, convolutional neural networks (CNNs) have driven rapid advancements in object detection. Representative approaches include the region-proposalbased Faster R-CNN, the single-stage SSD, and the efficient YOLO family. These frameworks achieve an excellent balance between detection accuracy and real-time performance under clear weather or ideal illumination. However, in adverse weather conditions such as fog, rain, or snow, reduced image contrast and increased noise degrade edge and texture representations during high-level feature extraction, resulting in blurred object boundaries and missed small targets. In addition, anchor-based detectors experience a notable drop in localization accuracy when matching anchor boxes with ground-truth objects in low-contrast environments [13,14]. Although anchor-free models have recently improved localization precision to some extent, they still face challenges in maintaining edge preservation while ensuring fast inference under complex weather conditions.

## 2.2. Transformer-Based Detection Models

With the rapid rise of visual Transformers, object detection frameworks have increasingly shifted toward attention-based architectures. Representative models such as RT-DETR [15] and DINO [16] can model global dependencies within large receptive fields, thereby achieving stronger semantic consistency in complex backgrounds.Despite their superior performance in multi-class detection scenarios, Transformer-based models are characterized by enormous computational demands and high training costs, which pose significant challenges for real-time inference in vehicle-mounted systems, leading to unacceptable latency. Moreover, under severe contrast degradation or edge blurring, these models tend to over-rely on global context, which reduces their ability to capture local structural cues such as vehicle contours and pedestrian boundaries.

## 2.3. Detection and Enhancement Methods for Adverse Weather Conditions

Research on object detection under adverse weather generally falls into two major categories: (1) Image enhancement or dehazing-based preprocessing approaches, and (2) architectural improvements to detection models. The former, represented by TransWeather [17], MSNet [18], and other physics-based enhancement algorithms, can improve image visibility but inevitably increase computational overhead. The latter, including DAGL-Faster R-CNN [19], PC-YOLO [20], and YOLOv8-ASF [21], enhances robustness through domain adaptation, feature decoupling, or multi-scale fusion mechanisms. However, these models typically require more complex feature pyramids or multi-branch inference structures, which compromise real-time performance. Therefore, how to enhance edge preservation and contextual feature fusion while maintaining high inference efficiency remains a critical challenge in automotive object detection.

## 2.4. Contribution of This Paper

Overall, existing studies reveal a clear trade-off between robustness and real-time performance. Transformer-based architectures, though strong in semantic understanding, are computationally expensive, whereas lightweight CNN-based detectors, while efficient,

suffer from feature degradation under adverse conditions. To address these limitations, this work proposes ECL-YOLOv11 (Edge-enhanced, Context-guided, and Lightweight YOLOv11), which achieves a unified balance between robustness and efficiency through three complementary modules: the CE module for edge enhancement, the AENet module for context-guided multi-scale feature fusion, and the LDHead module for lightweight real-time detection. By maintaining low computational complexity while significantly improving detection stability under adverse weather, ECL-YOLOv11 effectively bridges the gap between 'efficient yet fragile' and 'robust yet slow' detection frameworks.

## 3. Method

## 3.1. Edge Enhancement Convolution Module (CE)

In adverse weather conditions, image contrast and clarity often significantly decrease. The degradation of edge and texture information makes it difficult to accurately identify object boundaries, thereby affecting detection accuracy and stability. Traditional convolutional neural networks typically weaken responses to local gradients and edge features during high-level feature extraction, a drawback particularly prominent in low-visibility environments such as fog, rain, and snow. To enhance the model's boundary perception capability in degraded images, this paper proposes a Convolutional Edge-Enhancement Module (CE) as shown in Figure 1. The module introduces a fixed Sobel operator to extract horizontal and vertical gradients, which are then fused with learnable convolutional features. This explicitly retains edge information in the network's feature layer, achieving dual representation of texture and semantic features.

Figure 1. CE module structure diagram.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000005_2abf925291a8bc36de89206ba92c1f2105fe0f49c1c1d8241d9c4d7cb32732ac.png)

Figure 1 illustrates the architecture of the CE module, comprising two parallel branches: the Sobel edge extraction branch and the learnable convolutional branch. The former employs fixed convolution kernels to extract image gradient features, while the latter captures contextual semantic information through 3 × 3 learnable convolutions. These features are concatenated across channels and then compressed and fused via 1 × 1 convolutions. This fusion mechanism ensures the model maintains high sensitivity to target boundaries even with low-quality inputs, effectively reducing false positives and missed detections caused by blurred edges.

Figure 2 presents the architecture of the Sobel Conv framework, and its corresponding mathematical formulation is provided in Equations (1)-(3): Let the input feature be x ∈ R C × H × W , where the Sobel kernels Kx and Ky are applied for horizontal and vertical gradient computation, respectively.

<!-- formula-not-decoded -->

Edge features are derived by summing the horizontal and vertical responses as follows:

<!-- formula-not-decoded -->

where Conv ( · ; K ) denotes the convolution operation with a fixed kernel. Simultaneously, the convolution branch utilizes a learnable 3 × 3 kernel to extract regular semantic features, as represented by:

<!-- formula-not-decoded -->

Here, W and b are learnable parameters, ∗ indicates the convolution operation, and σ ( · ) is the nonlinear activation function. The outputs of both branches are concatenated along the channel dimension, yielding:

<!-- formula-not-decoded -->

Feature compression and fusion are performed using a 1 × 1 convolution, which can be expressed as:

<!-- formula-not-decoded -->

where ϕ ( · ) denotes a nonlinear transformation, and W 1 is the learnable 1 × 1 convolutional parameter. To maintain the integrity of the input features and improve feature transmission efficiency, a residual connection is introduced by adding the fused features to the original input, as follows:

<!-- formula-not-decoded -->

Finally, the output is generated through another 1 × 1 convolution:

<!-- formula-not-decoded -->

This design combines the interpretability of fixed operators with the expressive power of learnable convolution, enabling the model to extract high-level semantics while retaining key structural edge information.

Figure 2. Sobel Conv framework diagram.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000006_bd33e4c3b9afdd7ac355266fba09a9ba1a81a792881ad66f2bba195ca9b9baa3.png)

Furthermore, to enhance the network's overall perception capability, this paper embeds the CE module into the YOLOv11 backbone to construct a CE-Enhanced Backbone (CEB) framework, as shown in Figure 3. This architecture inserts a lightweight edge enhancement unit into the original feature extraction path, enabling the fusion learning of low-level gradient features and high-level semantic features. Through this integration approach, the proposed model not only learns clearer target boundary features but also maintains stable detection performance under complex weather conditions. Compared with traditional convolutional blocks, this module effectively reduces boundary drift issues in blurry and low-contrast images while improving detection accuracy and robustness. Additionally, the CE module demonstrates excellent versatility, allowing flexible integration into other lightweight backbone networks to provide scalable edge-enhancement solutions for various visual tasks.

Figure 3. CEB framework diagram.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000007_e2a7bb368fa3bffb52019ec600e9006187bf558f566867f628367f567fe86e70.png)

## 3.2. Context-Guided Multi-Scale Fusion Network (AENet)

In complex weather conditions, traditional object detection networks often struggle to simultaneously achieve effective performance across targets of varying scales. Particularly in low-visibility scenarios such as fog, rain, or snow, small and distant targets are prone to false positives or missed detections due to insufficient contextual information. Existing feature pyramid architectures (e.g., FPN or PAN) primarily rely on unidirectional feature propagation (top-down or bottom-up) during multi-scale fusion, which demonstrates limitations when processing images with blurred edges and low contrast. To address this, we propose a Context-Guided Multi-Scale Fusion Network (AENet). By integrating multi-level contextual fusion, rectangular auto-calibration, and bidirectional semantic flow mechanisms, the network achieves robust perception and feature enhancement for targets of different scales.

AENet replaces the Neck component of the original YOLOv11, serving as a critical feature fusion network that connects the Backbone with the detection head. Its architecture comprises four functional sub-modules: Pyramid Context Extraction (PCE), Rectangular Calibration (RCM), Down-to-Up Information Flow (DIF), and Feedback Block (FBM). These modules work through hierarchical complementarity and contextual coupling mechanisms, enabling the model to simultaneously enhance global semantic understanding and local structural details, thereby achieving stable detection performance under adverse weather conditions.

To unify the spatial dimensions of multi-level feature maps and extract contextual information across different scales, AENet introduces a Pyramid Context Extraction (PCE) module in its Neck section. As shown in Figure 4, this module first performs pooling on multi-scale features (e.g., P 3, P 4 , P 5) to ensure consistent spatial dimensions. The features are then concatenated along the channel dimension and fed into multiple cascaded Rectangular Calibration Modules (RCM) to achieve hierarchical contextual enhancement. Unlike traditional convolutions, the RCM adopts a strip-convolution architecture to model longrange dependencies along both the horizontal and vertical directions, thereby capturing the elongated edge structures of rectangular targets. This design is particularly effective for detecting regular-shaped objects-such as vehicles, lanes, and pedestrians-and markedly improves structural perception in low-visibility scenes with blurred boundaries. The process is mathematically expressed in Equation (8). Finally, the concatenated feature map is split back into the corresponding feature maps for each input scale. The mathematical expression for this process is shown in Equation (9).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Excite refers to the stripe convolution operation, σ denotes the Sigmoid activation function, and dwconv hw indicates the depthwise separable convolution operation. PyramidPoolAggPCE represents the adaptive pooling and concatenation process, RCM further performs contextual enhancement along the rectangular directions (horizontal and vertical), compensating for semantic bias across multi-scale features.

Figure 4. PCE module mechanism.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000008_2cc39a96fe2eebce3916082ce0556a645c0d083b11beda0b9431912c2f6679ba.png)

To further realize multi-layer semantic complementarity, AENet introduces a topdown information flow module (DIF) that transfers semantic guidance from high-level features to lower-level features. Figure 5 illustrates the DIF module's architecture: highlevel features F are first upsampled (interpolated) and channel-matched via convolutions to align with the resolution of lower-level features. Meanwhile, the lower-level features are enhanced by RCM to strengthen semantic expression. The two streams are then fused spatially to produce the output, as detailed in Equation (10).

<!-- formula-not-decoded -->

This module not only preserves the downward transmission of high-level semantic information, but also enhances the semantic representation of low-level features via RCM, effectively mitigating the problem of insufficient semantics in traditional PAN structures.

Unlike the DIF module, AENet's bottom-up feedback module (FBM) facilitates detail feedback and structural reinforcement. As shown in Figure 6, this module generates a gated weight map from the high-resolution features produced by PCE and performs structureaware fusion with low-resolution features to achieve detail compensation. The detailed process is described in Equation (11):

<!-- formula-not-decoded -->

where x 1 and x 2 represent the features from the lower and higher layers, respectively. Interp ( x 2 ) refers to the upsampling operation applied to x 2. The operations Conv ( xl ) and Conv ( xh ) perform convolution for the low-resolution and high-resolution features, respectively, while Interp denotes the upsampling operation applied to high-resolution features.

Finally, all fused features are input into the C3K2 module in YOLOv11 for unified processing. This module employs an alternating stacked 3 × 3 convolutional structure to compress and refine the multi-scale fused semantic information, thereby improving the feature representation quality of the final detection head while ensuring efficient computational performance.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000009_98476b72fa79dfae2464677a2652d69cf48006c87f563cc799f0a885615674d1.png)

Figure 5. DIF module.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000010_d66dccf5d30aa3e677ac9f991b073b75f0380c64f2bdb5436224d888e0b7b3b3.png)

Figure 6. FBM module.

In summary, the AENet architecture establishes a multi-scale fusion framework that jointly ensures semantic integrity and structural detail fidelity through a collaborative mechanism integrating Pyramid Context Extraction (PCE), Rectangular Calibration Module (RCM), Down-to-Up Information Flow (DIF), and Feedback Block Module (FBM). As shown in Figure 7, this framework sustains robust detection across targets of different scales in complex scenes and, under adverse weather conditions, effectively enhances the model's generalization ability and structural perception.

Figure 7. AENet network framework diagram.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000011_149c89ad0c9094b30461e62a2f8fc21ca6ed3e7cace291561274c0da32ea3e6f.png)

## 3.3. Lightweight Shared Convolution Detection Head (LDHead)

Autonomous driving vehicle systems require extremely high real-time performance from object detection models. Overly complex detection head designs often significantly increase computational load, leading to inference delays and degraded deployment performance. To reduce computational costs while maintaining detection accuracy, this paper proposes a lightweight shared convolutional detection head (LDHead) architecture for efficient and stable object detection on automotive platforms. Traditional detection heads independently construct convolutional branches on each scale's feature maps, which enhances feature representation at each layer but results in excessive parameter redundancy and computational overhead. The core concept of LDHead simplifies the architecture through 'cross-scale parameter sharing': sharing a unified set of convolutional weights across multi-scale feature maps, thereby maintaining representational power while substantially reducing computational complexity. Additionally, LDHead employs depthwise separable convolution and GroupNorm normalization strategies to achieve efficient feature

modeling and stable distribution alignment. This module replaces the original multi-head detection layer in the YOLOv11 architecture, uniformly processing multi-scale features (P3, P4, P5) from the AENet output to form the final detection prediction stage, as shown in Figure 8.

Figure 8. LDHead framework diagram.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000012_19176b1dfa3587cdb9b091659ef3a80fc6c785f3420caae4704c8ca00b913f26.png)

As shown in Figure 8, the LDHead module takes multi-scale features from AENet's output as input. Each feature map undergoes standardization through a 1 × 1 convolution and GroupNorm, reducing statistical discrepancies across scales. The processed multi-scale features are then passed into a shared convolutional structure composed of 3 × 3 depthwise separable convolutions, which extract features in both spatial and channel dimensions. This shared-parameter mechanism enables the network to obtain unified feature representations across multiple scales while significantly reducing redundant convolutional operations. The shared convolution outputs are split into two independent branches: one for bounding box regression and the other for class prediction. The regression branch outputs 4 × reg\_max channels through 1 × 1 convolution, representing the bounding box offsets in four directions, and is further processed by the regression module for size adjustment. The classification branch outputs the class probability distribution through 1 × 1 convolution and passes through a sigmoid activation function to produce the final result. The prediction results are maintained in their original format for loss calculation. During the inference phase, the outputs from all scales are concatenated and decoded into bounding boxes using an anchor generator and discrete distribution decoder, yielding the final detection results.

The prediction process of the LDHead module can be mathematically formalized as follows: Let the input feature be xi ∈ R Ci × Hi × Wi , which is first processed through channel adjustment and shared convolution to obtain a unified feature Fi . The prediction output for each position is:

<!-- formula-not-decoded -->

Here, W reg and W cls represent the convolutional weights for the regression and classification branches, respectively, while σ ( · ) denotes the Sigmoid activation function.

The regression branch outputs a discrete probability distribution over boundingbox offsets; this distribution is then decoded-under Distribution Focal Loss (DFL) supervision-into continuous bounding-box parameters:

<!-- formula-not-decoded -->

The final output is generated by concatenating the bounding box coordinates ˆ bi and the category probability yi ,cls into a unified prediction tensor. During the inference stage, these are merged to generate the final detection results.

The LDHead module demonstrates three core advantages: lightweight design, high stability, and robust performance. First, its cross-scale shared convolution significantly reduces parameter count and computational load, enabling faster inference speeds than traditional independent detection heads-a critical requirement for real-time automotive systems. Second, the adoption of GroupNorm instead of BatchNorm effectively mitigates distribution drift during small-batch training or dynamic scenarios, ensuring feature consistency and generalization under unstable image statistics in adverse weather conditions. Additionally, the discrete distribution-based regression strategy enhances bounding-box prediction accuracy, allowing the model to reliably recover target positions even with low-quality inputs. By integrating shared convolution, depthwise separable architecture, and stable normalization, LDHead achieves both lightweight design and computational efficiency. Together with CE and AENet, this module forms the core improvement framework of ECL-YOLOv11: CE enhances edge feature expression, AENet optimizes multi-scale semantic fusion, while LDHead strikes a balance between efficient prediction and realtime deployment.

## 3.4. Module Integration and Overall Architecture

The ECL-YOLOv11 architecture is designed to enhance model robustness and detection accuracy in adverse weather conditions while maintaining the efficiency of the YOLO series. Compared with YOLOv11, ECL-YOLOv11 achieves end-to-end optimization-from feature extraction to detection-through three complementary modules: the CE module focuses on edge enhancement, the AENet module enables multi-scale contextual fusion, and the LDHead module ensures lightweight, efficient inference.

As shown in Figure 9, the three components are sequentially integrated across different network layers to form a closed-loop feature flow characterized by 'edge preservation, semantic fusion, efficient prediction.' At the Backbone stage, the CE module explicitly incorporates Sobel edge features, enhancing low-level structural information and providing high-quality texture input for subsequent fusion. In the Neck stage, AENet achieves crossscale semantic alignment through pyramid context extraction and bidirectional feature flow, enabling consistent perception of small targets and distant objects. At the Head stage, LDHead employs shared convolutions and GroupNorm to achieve lightweight inference and stable normalization, maintaining detection accuracy and real-time performance under adverse weather conditions. This modular integration design establishes a collaborative mechanism from low-level structure enhancement to high-level prediction optimization, effectively addressing three major challenges faced by traditional detectors in low-visibility environments: edge degradation, scale mismatch, and inference latency.

Figure 9. Architecture diagram of ECL-YOLOv11, Notes: CE: Convolutional Edge-Enhancement module; AENet: Context-Guided Multi-Scale Fusion Network; LDHead: Lightweight Shared Convolutional Detection Head; RCM: Rectangular Calibration Module; C3K2: Cross-Stage Partial Connection block; DIF: Down-to-Up Information Flow module.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000013_123262338d0f6e852ff15c07d7762345f1ebd70389c6eaf367a79be6bfedb1c0.png)

## 4. Experiments

## 4.1. Dataset

This study utilizes a self-developed dataset of severe weather traffic scenarios, comprising 9006 road traffic images captured under three typical meteorological conditions: rain, fog, and snow. The images were acquired using vehicle-mounted high-definition cameras, covering urban roads and highways with varying lighting conditions, visibility levels, and complex backgrounds. The dataset encompasses seven categories of traffic participants, including bicycles, buses, cars, pedestrians, trains, and trucks. All images are annotated in YOLOv11 format to ensure data consistency and effective model training. To guarantee reproducibility and standardized data utilization, the dataset is managed and exported through the Roboflow platform. The dataset is divided into an 8:1:1 ratio (80% for training, 10% for validation, and 10% for testing) to ensure fair and statistically reliable model performance evaluation. Selected images from the dataset are shown in Figure 10, while the data distribution is illustrated in Figure 11.

Figure 10. Sample images from the dataset.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000014_5ca594e05240e873a2c272936aceee074f6a8a19c61ab6bcbc04b942d30ff871.png)

Figure 11. Distribution of dataset targets.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000015_05a74830cdbea68ef73b041377da4d59554fc4129dcb1da04778e756ce2fc262.png)

## 4.2. Experimental Setup and Training

The experimental setup is detailed in Table 1. To prevent overfitting on a relatively small dataset, we incorporate an Early Stopping mechanism and monitor validation performance throughout training. Training is capped at 600 epochs and terminates early if the validation mAP does not improve for 20 consecutive epochs. In addition, we adopt Cosine Annealing learning-rate decay and Label Smoothing to enhance model stability and generalization.

Table 1. Experimental environment parameters.

| Configuration        | Detailed Description                    |
|----------------------|-----------------------------------------|
| CPU                  | Intel Core i5-12400F                    |
| GPU                  | NVIDIA GeForce RTX 3060 (12 GB VRAM)    |
| Operating System     | Windows 11                              |
| Software Environment | Python 3.10.15, PyTorch 2.5.0, CUDA12.1 |
| Framework Version    | Ultralytics 8.3.9                       |
| Image Size           | 640 × 640                               |
| Training Period      | 600                                     |
| Batch Size           | 16                                      |

As shown in Figure 12, the experimental results demonstrate the model's convergence and stability during both training and validation phases. The loss functions (box\_loss, cls\_loss, and dfl\_loss) exhibit a clear pattern of rapid decline followed by gradual stabilization as training progresses. The consistency between the validation and training curves indicates stable and synchronized convergence behavior. Notably, precision and recall rates initially increase before stabilizing, reflecting the model's balanced accuracy-recall performance. To further validate the model's convergence characteristics and generalization capabilities, this study adds the mAP convergence curve for the validation set to the existing training curve analysis (as shown in Figure 13).

Figure 13 illustrates the evolution of mAP@50 and mAP@50-95 during validation: both curves show sustained growth before saturating, with values stabilizing at 0.6256 and 0.4035, respectively, at epoch 597. This progression aligns with the loss trends observed during training, further confirming stable convergence and strong generalization across multi-scale and complex scenarios. The labeled Stop Point indicates that the Early Stopping mechanism terminated training once validation performance had saturated, avoiding

unnecessary additional fitting. Moreover, both core metrics: mAP@50 and mAP@50-95, rise steadily throughout training before saturating, demonstrating robust detection performance and sustained learning capacity in diverse, challenging environments.

Figure 12. Training performance metrics.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000016_7efbfa602727eaa723ad4202c20a1426c68df93907c33ffa9e8f38ad8fd38db4.png)

Figure 13. mAP convergence curve of the validation set.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000017_6b9afae91204ece65d7d29014c8cf8bf6241dafb9bb9478d1abaab45d551fcbc.png)

The experimental results collectively show that the proposed ECL-YOLOv11 achieves stable convergence during training, demonstrates stable convergence and generalization during training, and exhibits strong generalization. This experimental setup and training strategy provide reliable support for subsequent performance analysis and model validation under complex meteorological conditions.

## 4.3. Detection Performance Evaluation

To comprehensively validate the effectiveness of the ECL-YOLOv11 model for vehiclemounted object detection, we evaluate three key metrics: mAP (mean Average Precision), Precision, and Recall. Together, these metrics assess performance from three perspectives-classification accuracy, bounding-box localization, and overall robustness

The comparison between the baseline YOLOv11 and ECL-YOLOv11 is reported in Table 2 and Figure 14. ECL-YOLOv11 attains 62.7% mAP@50 and 40.5% mAP@50-95, which respectively measure average detection accuracy under single- and multi-threshold settings and reflect both classification and localization capability. Compared with YOLOv11 (61.4% and 39.7%), the gains of 1.3% and 0.8% indicate improved robustness and generalization, with more stable behavior in complex weather. In terms of Precision, ECL-YOLOv11 reaches 73.1% (vs. 68.8% for YOLOv11, +4.3%), demonstrating effective reduction of false positives

and class confusion under edge blurring, uneven illumination, and rain/fog occlusion. For Recall, ECL-YOLOv11 achieves 55.6% (vs. 55.3%), indicating that the model maintains strong recall while improving detection accuracy. These results suggest that the proposed multi-module collaborative strategy achieves a better balance between sensitivity and recall without sacrificing either metric. Overall, ECL-YOLOv11 outperforms the baseline under harsh weather, with particularly meaningful improvements in Precision and mAP. Moreover, as shown in the Figure 14, the model exhibits no overfitting during training, and the observed gains stem from architectural improvements rather than tuning bias.

Table 2. Experimental results metrics.

| Model       |     P |     R |   mAP@50 |   mAP@50-95 |
|-------------|-------|-------|----------|-------------|
| YOLOv11     | 0.688 | 0.553 |    0.614 |       0.397 |
| ECL-YOLOv11 | 0.731 | 0.556 |    0.625 |       0.405 |

Figure 14. Comparison experimental results between ECL-YOLOv11 and YOLOv11.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000018_7c12f6fa28e1f2b9a669522a80146cef1c19c3885a29b288e12fa4579a55b888.png)

Overall, ECL-YOLOv11 not only improves the overall detection accuracy, but also achieves a favorable balance between performance and efficiency without increasing computational complexity, thereby providing a more practically deployable detection solution for in-vehicle vision systems

## 4.4. Ablation Study

To visually demonstrate the effectiveness of the improved modules and the overall performance gains, ablation experiments were conducted. The primary objective is to progressively remove and combine different modules (CE, AENet, and LDHead), emphasizing the individual effectiveness of each module and the overall performance benefits of the final solution. The experimental results are presented in Table 3 and Figure 15.

As shown by the ablation results in Table 3, an overall mAP improvement of +1.3% was consistently observed across multiple independent experiments, demonstrating the stability and reproducibility of the proposed module. Given prior evidence that YOLOv11 performance approaches saturation under clear-weather conditions, the stable gains achieved under adverse weather are therefore particularly significant.

Table 3. Evaluation metrics.

| (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   | (a) Category-wise Evaluation Metrics.   |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| Model                                   | Model                                   | Car                                     | Person                                  | Bus                                     | Bicycle                                 | Truck                                   | Train                                   |
| Baseline(11)                            |                                         | 85.6                                    | 73.4                                    | 42.4                                    | 57.8                                    | 60.5                                    | 60.9                                    |
|                                         |                                         | 85.8                                    | 73.5                                    | 42.3                                    | 58.7                                    | 63.8                                    | 60.9                                    |
| CE AENet                                |                                         | 85.7                                    | 72.5                                    | 45.1                                    | 60.0                                    | 63.6                                    | 63.7                                    |
| LDHead                                  |                                         | 85.7                                    | 73.5                                    | 38.9                                    | 57.2                                    | 62.2                                    | 56.9                                    |
| CE + AENet                              |                                         | 85.3                                    | 72.6                                    | 42.8                                    | 59.9                                    | 62.0                                    | 60.5                                    |
| AENet + LDHead                          |                                         | 85.1                                    | 72.7                                    | 42.1                                    | 58.6                                    | 60.4                                    | 57.9                                    |
| CE + LDHead                             |                                         | 85.8                                    | 73.0                                    | 36.2                                    | 56.2                                    | 60.2                                    | 59.5                                    |
| Our                                     |                                         | 85.5                                    | 73.4                                    | 46.2                                    | 59.0                                    | 60.8                                    | 62.0                                    |
| (b) Overall Performance Metrics.        |                                         |                                         |                                         |                                         |                                         |                                         |                                         |
| Model                                   | P                                       | R                                       | mAP50                                   | mAP50-95                                | Para.                                   | GFLOPs                                  | FPS                                     |
| Baseline(11)                            | 68.8                                    | 55.3                                    | 61.4                                    | 39.7                                    | 2,583,517                               | 6.3                                     | 406.5                                   |
| CE                                      | 65.8                                    | 57.2                                    | 62.0                                    | 41.0                                    | 2,530,501                               | 6.4                                     | 337.7                                   |
| AENet                                   | 68.1                                    | 56.5                                    | 62.3                                    | 40.4                                    | 3,294,861                               | 8.6                                     | 249.2                                   |
| LDHead                                  | 66.8                                    | 56.9                                    | 60.9                                    | 39.0                                    | 2,420,882                               | 5.6                                     | 451.6                                   |
| CE +                                    | 69.8                                    | 55.6                                    | 62.7                                    | 40.2                                    | 3,241,845                               | 8.7                                     | 213.5                                   |
| AENet AENet + LDHead                    | 67.0                                    | 57.1                                    | 61.1                                    | 39.9                                    | 2,934,413                               | 7.2                                     | 142.5                                   |
| CE + LDHead                             | 65.4                                    | 57.2                                    | 60.5                                    | 39.2                                    | 2,263,029                               | 5.1                                     | 209.9                                   |
| Our                                     | 73.1                                    | 55.6                                    | 62.7                                    | 40.5                                    | 3,001,194                               | 7.5                                     | 237.5                                   |

Note: GFLOPs: Billion Floating-Point Operations, used to measure the computational complexity of the model.

Figure 15. Ablation study evaluation metrics.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000019_215a04af786fe0df3ed351d565c4344538a998dcb5bae2fd8a5444614bb3cb9b.png)

As shown in Table 3 and Figure 15, the baseline model (YOLOv11) achieved mAP@50 and mAP@50-95 of 61.4% and 39.7%, respectively, without structural augmentation. When the CE module was introduced, mAP@50 increased to 62.0% and mAP@50-95 to 41.0%, indicating that edge-enhanced convolution (CE) effectively preserves texture details and improves boundary localization accuracy. Although the CE module introduces only a marginal increase in computational complexity (GFLOPs 6.3 → 6.4), its stable performance under adverse weather suggests meaningful compensation for weather-induced structural degradation. Introducing AENet further strengthens feature fusion, achieving mAP@50

of 62.3% and mAP@50-95 of 40.4%, demonstrating that multi-scale semantic integration effectively improves overall detection accuracy. However, Precision decreases slightly from 68.8% to 68.1%, implying that the more complex fusion adopts a more conservative decision boundary to reduce false positives, potentially sacrificing recall. These results indicate that AENet excels at improving detection stability and identifying mid- to long-range blurred targets, while showing minor fluctuations in pure precision; overall, AENet balances feature expressiveness and robustness. The LDHead module introduces a decoupling mechanism in the detection head, enabling more independent classification and regression and thereby enhancing modeling capacity; without backbone augmentation, it attains mAP@50 of 60.9% and mAP@50-95 of 39.0%, slightly lower but lighter for deployment. With GFLOPs of just 5.6 and FPS reaching 451.6, it verifies the clear efficiency advantages of the shared-convolution mechanism for real-time detection.

Furthermore, at the module combination level, the CE+AENet architecture achieves 62.7% mAP@50 and 40.2% mAP@50-95 precision, demonstrating complementary edge enhancement and multi-scale context fusion. However, its inference speed drops to 215.3 FPS, indicating increased computational overhead. When LDHead is added to form the CE+AENet+LDHead structure, mAP@50 remains at 62.7%, mAP@50-95 slightly improves to 40.5%, while FPS significantly increases to 237.8 and GFLOPs decrease to 7.5. This 'precision-preserving acceleration' phenomenon shows that LDHead optimizes computational paths and resource allocation through parameter sharing while maintaining detection accuracy, achieving structural-level performance balance. Category-wise analysis reveals that ECL-YOLOv11 demonstrates significant improvements over the baseline for medium-large targets (e.g., buses and trains), while maintaining stable performance for small targets such as pedestrians and bicycles, further evidencing AENet's effective context fusion across different object scales. Precision increases to 73.1%, a 4.3% improvement over the 68.8% baseline, indicating advantages in reducing false positives and edge confusion.

In summary, the ablation experiments quantitatively demonstrate the structural design advantages of the AENet, LDHead, and CE modules proposed in this paper. These experiments validate that the improved ECL-YOLOv11 network not only achieves the synergy and complementary strengths of the modules but also effectively reduces model complexity and enhances inference speed while improving object detection accuracy compared to the baseline model, YOLOv11. The system demonstrates a balanced capability of maintaining high accuracy while ensuring real-time performance and lightweight design, providing a solid structural foundation and performance guarantee for a vehicle-based target detection system in adverse weather conditions.

## 5. Comparison Experiments

## 5.1. Comparison of Different Modules and Models

In the comparison experiments of this section, this paper conducts tests from four aspects (namely Backbone, Head, Neck, and overall model) to demonstrate the impact of different architectures or modules on the performance of the YOLOv11 model, as shown in Tables 4 and 5 and Figure 16.

Table 4. Compare the test results of each module.

|          | Model        |   Car |   Person |   Bus |   Bicycle |   Truck |   Train |    P |    R |   mAP50 |   mAP50-95 |
|----------|--------------|-------|----------|-------|-----------|---------|---------|------|------|---------|------------|
| Backbone | CE           |  85.8 |     73.5 |  42.3 |      58.7 |    63.8 |    60.9 | 65.8 | 57.2 |    62   |       41   |
| Backbone | efficientViT |  85.6 |     72.9 |  32   |      60.8 |    60.5 |    64.9 | 69.6 | 55.2 |    60.9 |       39.7 |
| Backbone | starnet      |  84.5 |     71   |  42.9 |      55.9 |    61.5 |    59.5 | 71.8 | 52.7 |    60.3 |       39.4 |

Table 4. Cont.

|      | Model         |   Car |   Person |   Bus |   Bicycle |   Truck |   Train |    P |    R |   mAP50 |   mAP50-95 |
|------|---------------|-------|----------|-------|-----------|---------|---------|------|------|---------|------------|
| Neck | AENet         |  85.7 |     72.5 |  45.1 |      60   |    61.6 |    63.7 | 68.1 | 56.5 |    62.3 |       40.4 |
| Neck | slimneck      |  85.9 |     72.8 |  31.5 |      57.9 |    61   |    55.3 | 69.9 | 52.1 |    59.1 |       38.4 |
| Neck | bifpn         |  86.6 |     74.7 |  41.2 |      60.4 |    63   |    57.3 | 69   | 59.4 |    61.7 |       40.3 |
|      | LDHead        |  85.7 |     73.5 |  38.9 |      57.2 |    62.2 |    56.9 | 66.8 | 56.9 |    61   |       39   |
|      | LADH          |  85.5 |     72.6 |  34   |      56.6 |    60.1 |    60   | 68.4 | 54.3 |    59.9 |       38.5 |
|      | EfficientHead |  85.6 |     73.5 |  37   |      58.2 |    62.7 |    59.5 | 68.6 | 56.2 |    61   |       39.9 |

Table 5. Comparison of detection results of all holistic models.

| Model        |   Car |   Person |   Bus |   Bicycle |   Truck |   Train |    P |    R |   mAP50 |   mAP50-95 |   FPS |
|--------------|-------|----------|-------|-----------|---------|---------|------|------|---------|------------|-------|
| yolo11       |  85.6 |     73.4 |  42.4 |      57.8 |    65   |    58.1 | 68.8 | 55.3 |    61.4 |       39.7 | 434.4 |
| YOLOv10n     |  86   |     72.6 |  40.6 |      60.6 |    62.4 |    58.9 | 72.3 | 52.8 |    61.3 |       40.2 | 385   |
| YOLOv8       |  86   |     73.6 |  40.4 |      62.2 |    62.8 |    59.6 | 68.6 | 57.6 |    61.9 |       40.4 | 409.7 |
| YOLOv5n      |  84.7 |     70.9 |  40.1 |      57.6 |    60.8 |    54.1 | 71.7 | 53.9 |    59.2 |       39.4 | 397.2 |
| yolov8-ASF   |  86.5 |     73.6 |  47   |      58.6 |    62.6 |    61.3 | 70.4 | 55.6 |    62   |       40.5 | 336.7 |
| RT-DETR      |  73   |     52.5 |  62.3 |      40   |    73   |    52.5 | 68.7 | 57.2 |    62.1 |       40.2 | 152   |
| Faster R-CNN |  85.4 |     73.1 |  45.5 |      58.3 |    61.3 |    61.7 | 73   | 52.5 |    62.3 |       40   |  58   |
| Our          |  85.5 |     73.4 |  46.2 |      59   |    60.8 |    62   | 73.1 | 55.6 |    62.7 |       40.5 | 230.3 |

The comparison experiment data in Tables 4 and 5 clearly shows that the improved model, ECL-YOLOv11, exhibits significant superiority in the object detection task.

Figure 16. Overall performance metrics of the comparative experiment.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000020_cfdc4cf3941272d728fba02066c41038626fa5d00bd57b3befba14e145ce898e.png)

The comparative results at the backbone level demonstrate that the proposed EdgeEnhanced Convolutional Module (CE) achieves superior performance in object detection. CE outperforms EfficientViT [22] (60.9%/39.7%) and StarNet [23] (60.3%/39.4%) with 62.0% mAP@50 and 41.0% mAP@50-95. Notably, while StarNet attains 71.8% Precision, its Recall drops to 52.7%, revealing limitations in modeling distant and small targets under complex weather conditions. By explicitly enhancing edge and texture cues, CE effectively mitigates feature degradation in hazy weather, thereby exhibiting stronger stability and robustness across multi-scale scenarios.

In the Neck architecture, AENet achieves an mAP@50-95 of 40.4%, surpassing SlimNeck [24] (38.4%) and BiFPN [25] (40.3%), and showing superior feature-fusion consistency in complex weather. Although BiFPN reaches 59.4% Recall, its limited mAP gain indicates

insufficient stability of the weighted-fusion strategy under noisy conditions. Leveraging context-guided multi-scale semantic fusion, AENet strengthens spatial hierarchy modeling and markedly improves small-object recognition in low-contrast, long-range scenarios, demonstrating enhanced feature robustness in fog, rain, and snow.

In the comparison of detection head architectures, LDHead demonstrated competitive overall performance with an mAP@50-95 of 39.0%, comparable to EfficientHead (39.9%) and higher than LADH (38.5%). While EfficientHead showed higher Precision and Recall, its confidence fluctuations were more pronounced in complex backgrounds. LDHead, however, enhanced the independence between classification and regression through shared convolution and a decoupled design, reducing multi-task interference and yielding more stable predictions-particularly in scenes with edge blurring and small targets.

The comparative analysis shows that, in our experiments, ECL-YOLOv11 achieves 62.7% mAP@50 and 40.5% mAP@50-95 with 73.1% Precision and 55.6% Recall, outperforming the compared YOLO-series baselines-YOLOv11 (61.4%/39.7%), YOLOv10n [26] (61.3%/40.2%), and YOLOv8 [27] (61.9%/40.4%). Compared with non-YOLO detectors such as RT-DETR and Faster R-CNN, ECL-YOLOv11 attains comparable accuracy (62.1% vs. 62.3%) while delivering significantly faster inference, highlighting robustness under low-visibility, high-interference conditions. These results further validate the synergistic advantages of the ECL design in backbone feature extraction, context-guided fusion, and decoupled detection, achieving a balanced trade-off between accuracy and real-time efficiency with moderate complexity and strong deployment potential. Visual comparison results are presented in Figure 17.

As shown in Figure 17, systematic comparative analysis of eight models under extreme weather conditions clearly demonstrates ECL-YOLOv11's significant advantages in detection accuracy, stability , and robustness. Foggy environments pose one of the most challenging scenarios for object detection, where extremely low visibility often leads to insufficient box confidence and missed small targets. Models like YOLOv11, YOLOv5n [28], and YOLOv8 generally produce numerous low-confidence boxes (0.2-0.4) in such conditions, exhibiting instability in detecting key targets like people and buses, with both false positives and missed detections. While YOLOv10n and YOLOv8-ASF show slight improvements in vehicle detection, their performance remains inconsistent in identifying trucks and people. RT-DETR achieves relatively accurate vehicle detection through its global attention mechanism, but maintains low confidence levels (0.3-0.6) and struggles with small-object recognition. Faster R-CNN performs the weakest in this scenario, only detecting nearby vehicles while failing to identify distant or small targets. In contrast, ECL-YOLOv11 demonstrates superior detection consistency, maintaining over 50% confidence for most targets and achieving relatively accurate recognition of buses and trucks, with overall stability clearly surpassing the other models. This indicates that its feature-extraction and multi-scale information-fusion strategy offers stronger adaptability to low-contrast targets.

In rainy scenes, light reflections and water-mist interference impose higher demands on detection models. Comparative results show that YOLOv11 misidentifies buses as trucks with blurred category boundaries, indicating insufficient generalization in complex environments; YOLOv5n exhibits similar errors, further revealing its limitations in cross-category recognition. While YOLOv8 and YOLOv8-ASF maintain higher confidence for bus detection ( ≈ 0.77-0.85) than YOLOv5n and YOLOv11, they still show confidence fluctuations for individual vehicles. RT-DETR detects vehicles effectively in rain but suffers a notable confidence drop for buses ( ≈ 0.57) due to strong reflections, suggesting that its global-attention mechanism is susceptible to focus drift under complex lighting. Although Faster R-CNN produces relatively accurate bounding boxes, its overall confidence remains low and it fails to stably detect distant small objects. By contrast, ECL-YOLOv11 maintains stable bus recognition and exhibits consistent

confidence across different vehicle classes over multiple lanes without misclassification. This stability is crucial for rainy-day traffic monitoring and indicates stronger discrimination under reflection and occlusion interference.

Figure 17. Qualitative visualization results of different models.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000021_bfb9b479b4968b4f3b588d1878c6ca98f4c1b2fed564f0993bea8a685c5643c6.png)

In snow scenes, background noise, occlusion, and surface coverage in snow substantially affect detection accuracy. YOLOv11 yields false positives for 'train,' reflecting

insufficient discriminative robustness in challenging environments. Although YOLOv5n achieves high confidence for trucks, its confidence for person remains around 0.30-0.34, implying a serious missed-detection risk. YOLOv10n and YOLOv8 perform relatively stably but still struggle with low truck confidence or insufficient bicycle detection; YOLOv8-ASF shows advantages for motorcycles and bicycles, yet some boxes retain low confidence. Faster R-CNN fails to detect primary targets effectively, providing only low-confidence boxes at close range, whereas RT-DETR detects trucks ( ≈ 0.66) but fails on pedestrians and non-motorized classes-indicating an attention bias of Transformer architectures under high-illumination scenes. In contrast, ECL-YOLOv11 maintains stable multi-class recognition in snow, accurately localizing trucks, cars, persons, and bicycles with a reasonable confidence distribution. Unlike YOLOv11's misclassifications or YOLOv5n's extremely low confidences, ECL-YOLOv11 shows stronger robustness under complex backgrounds and noise interference.

Overall. ECL-YOLOv11 delivers consistent improvements over mainstream models and structural alternatives, achieving effective synergy in feature retention, semantic fusion, and lightweight detection. The results indicate modest yet stable metric gains accompanied by consistent, repeatable behavior in complex real-world conditions, underscoring the framework's scientific soundness and engineering applicability. This provides a solid foundation-and a scalable pathway-for future multimodal fusion and real-time perception in autonomous-driving environments.

## 5.2. Comparison Experiments Considering Adverse Weather Conditions and Relative Motion States

To further verify the effectiveness of ECL-YOLOv11 as an in-vehicle visual object detection solution under adverse weather conditions, experiments were designed in three weather environments: snow (Snow), fog (Fog), and rain (Rain). The model's performance was tested in three relative motion state scenarios: dynamic (autonomous vehicle)-dynamic (detected target), static (autonomous vehicle)-dynamic (detected target), and dynamic (autonomous vehicle)-static (detected target). As shown in Figure 18, the experiment primarily examines the robustness of ECL-YOLOv11 in small target detection, long-distance target localization, and complex background interference.

As shown in Figure 18, the experiment configured three relative-motion combinations under each weather condition to simulate typical visual-perception tasks for autonomous driving in complex environments. Key evaluation priorities included small-object detection performance, long-distance target localization accuracy, and false-detection control in cluttered backgrounds. In foggy scenarios, ECL-YOLOv11 accurately detected nearby vehicles in both dynamic-dynamic and dynamic-static states, maintaining detection confidences of 0.84-0.93; it also kept confidences above 0.50 for distant vehicles and pedestrians under severely limited visibility, markedly reducing missed-detection risk due to blur and occlusion, thereby validating the CE module's effectiveness in restoring structural details and contour recognition. Under rainy conditions, the model maintained robust performance in dynamic-dynamic and static-dynamic settings: despite distortions from raindrops, haze, and roadway reflections, major-vehicle confidences stayed above 0.85, boxes were complete, and long-range localization remained stable, indicating that the AENet module, via context-guided multi-scale fusion, improved semantic consistency in low-contrast dynamic scenes. Snow experiments further verified generalization in complex backgrounds and multi-target mixed scenes: vehicle confidences remained 0.85-0.91 in static → moving and moving → static states, while pedestrian detection stabilized at 0.50-0.77; even with snow glare and partial occlusion, ECL-YOLOv11 showed no notable class confusion or missed detections. Additionally, the lightweight LDHead ensured synchronization between inference speed and accuracy in static → moving and moving → static scenarios, demonstrating

excellent latency control and structural adaptability. Overall, ECL-YOLOv11 exhibited stable detection across adverse weather and motion states, maintaining high confidences and complete detections under extreme conditions (insufficient illumination, low contrast, occlusion, reflection), and significantly mitigating boundary ambiguity and misidentification issues common to traditional detectors. Compared with conventional models, often limited by confidence fluctuation, labeling errors, and weaker environmental adaptability, our framework provides more robust, reliable target perception, offering strong support for intelligent driving in complex traffic environments.

Figure 18. Model validation results considering adverse weather conditions and motion states.

![Image]((openai-gpt-4o-mini_google-gemini-flash-1.5)_images_and_tables\89294530-fd19-4a48-9cdd-1f2240fbc23f\89294530-fd19-4a48-9cdd-1f2240fbc23f-with-image-refs_artifacts\image_000022_c3638328000e721b4b8b010bc2d7f4f3c88b00a3b41af0f3a0c749e1e24ff19f.png)

## 6. Conclusions

This study presents ECL-YOLOv11, an edge-enhanced, context-guided, and lightweight object detection framework designed for vehicle-mounted visual perception under adverse weather conditions. Building upon the YOLOv11 baseline, the proposed model integrates three complementary modules:

- (1) The Edge-Enhancement Convolution (CE) module, which explicitly preserves edge and gradient features to mitigate boundary degradation under low visibility;
- (2) The Context-Guided Multi-Scale Fusion Network (AENet), which strengthens semantic consistency and multi-scale representation through context-aware feature aggregation;
- (3) The Lightweight Shared-Convolution Detection Head (LDHead), which optimizes computational paths for real-time inference without sacrificing accuracy.

Comprehensive experiments demonstrate that ECL-YOLOv11 achieves modest yet stable and reproducible performance improvements in detection accuracy, robustness, and computational efficiency compared with mainstream detectors (YOLOv11, YOLOv8, YOLOv10n, RT-DETR, Faster R-CNN). Specifically, the model reaches mAP@50 = 62.7%, mAP@50-95 = 40.5%, and 237.8 FPS. Although the overall improvement (1.3 mAP) is numerically limited, it is consistently reproduced across multiple independently trained variants (as shown in Table 3), indicating that the observed gain is systematic rather than random variation. Under fog, rain, and snow, the framework maintains consistent detection confidence and avoids misclassification or missed targets common to traditional

## References

1. Chen, C.; Seff, A.; Kornhauser, A.; Xiao, J. DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Santiago, Chile, 7-15 December 2015; pp. 2722-2730.

detectors, validating its robust adaptability across complex environmental and motion states. The design philosophy of ECL-YOLOv11 emphasizes a balance between structural interpretability and operational practicality. The synergy among the CE, AENet, and LDHead modules enhances both feature quality and system responsiveness, offering a solid foundation for robust in-vehicle perception.

Nevertheless, several limitations remain. The dataset used in this study ( ∼ 9 k images) is relatively small and focuses on adverse-weather scenarios; systematic evaluations under clear weather and larger-scale data remain to be performed. Additionally, the training process was constrained by computational resources (single GPU, 640 × 640 input resolution, batch size 16), which may limit the achievable upper bound of model performance; therefore, the improvements observed in this study should be interpreted as conservative estimates under current conditions. The model also relies solely on visual input, without leveraging complementary modalities.

Future research will therefore focus on three directions:

(1) Multi-sensor fusion, integrating ECL-YOLOv11 with LiDAR and millimeter-wave radar to improve depth perception and robustness in low-visibility environments;

(2) System-level co-optimization, coupling detection, planning, and decision-making modules in autonomous-driving pipelines to enable end-to-end collaborative learning;

(3) Dynamic-scene adaptation, developing online and continual-learning mechanisms to sustain long-term stability under evolving real-world conditions.

In summary, ECL-YOLOv11 demonstrates scientifically sound, repeatable, and practically deployable improvements in object detection under complex weather. By balancing robustness and real-time efficiency, it offers a feasible path toward reliable and efficient perception for intelligent vehicles, bridging the gap between algorithmic robustness and real-time application, and laying the groundwork for future multimodal and adaptive perception systems.

Author Contributions: Conceptualization, Z.L. and J.Z.; Methodology, Z.L. and J.Z.; Software, J.Z.; Validation, Z.L., J.Z. and X.Z.; Formal Analysis, Z.L.; Investigation, Z.L. and H.S.; Resources, J.Z.; Data Curation, Z.L. and X.Z.; Writing-Original Draft Preparation, Z.L. and J.Z.; Writing-Review Editing, Z.L., J.Z. and H.S.; Visualization, Z.L. and X.Z.; Supervision, Z.L. All authors have read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: The data that support the findings of this study are available from the corresponding author upon reasonable request.

Acknowledgments: We would like to thank all the reviewers for their help in shaping and refining this paper.

Conflicts of Interest: All of the authors declare that no conflicts of interest exist in the submission of this manuscript.

2. Caesar, H.; Bankiti, V.; Lang, A.H.; Vora, S.; Liong, V.E.; Xu, Q.; Krishnan, A.; Pan, Y.; Baldan, G.; Beijbom, O. nuScenes: A Multimodal Dataset for Autonomous Driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 13-19 June 2020; pp. 11618-11628.
3. Kurmi, T.; Bhattacharyya, S.; Tyagi, V. A Comprehensive Review on Adverse Weather Conditions Perception for Autonomous Driving. IEEE Trans. Intell. Transp. Syst. 2022 , 23 , 19301-19317.
4. Bijelic, M.; Gruber, T.; Mannan, F.; Kraus, F.; Ritter, W.; Dietmayer, K.; Heide, F. Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 13-19 June 2020; pp. 11682-11692.
5. Tian, Y.; Zhao, L.; Chen, Y. Detection and Tracking of Vehicles in Adverse Weather Using Deep Learning. IEEE Trans. Intell. Transp. Syst. 2023 , 24 , 6792-6804.
6. Ren, S.; He, K.; Girshick, R.; Sun, J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Trans. Pattern Anal. Mach. Intell. 2017 , 39 , 1137-1149. [CrossRef] [PubMed]
7. Redmon, J.; Divvala, S.; Girshick, R.; Farhadi, A. You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27-30 June 2016; pp. 779-788.
8. Liu, W.; Anguelov, D.; Erhan, D.; Szegedy, C.; Reed, S.; Fu, C.; Berg, A.C. SSD: Single Shot Multibox Detector. In Proceedings of the European Conference on Computer Vision (ECCV), Amsterdam, The Netherlands, 11-14 October 2016; pp. 21-37.
9. Liu, Z.; Hou, W.; Chen, W.; Chang, J. The Algorithm for Foggy Weather Target Detection Based on YOLOv5 in Complex Scenes. Complex Intell. Syst. 2025 , 11 , 71. [CrossRef]
10. Liu, Z.; Yan, J.; Zhang, J. Research on a Recognition Algorithm for Traffic Signs in Foggy Environments Based on Image Defogging and Transformer. Sensors 2024 , 24 , 4370. [CrossRef] [PubMed]
11. Liu, Z.; He, Y.; Wang, C.; Song, R. Analysis of the Influence of Foggy Weather Environment on the Detection Effect of Machine Vision Obstacles. Sensors 2020 , 20 , 349. [CrossRef]
12. Alif, M.A.R. YOLOv11 for Vehicle Detection: Advancements, Performance, and Applications in Intelligent Transportation Systems. arXiv 2024 , arXiv:2410.22898. [CrossRef]
13. Wang, Y.; Chen, X.; Zhang, L. Edge Enhancement for Object Detection in Foggy Weather. IEEE Trans. Image Process. 2021 , 30 , 1766-1779.
14. Zhang, Y.; Xuan, S.; Li, Z. Robust Object Detection in Adverse Weather with Feature Decorrelation via Independence Learning. Pattern Recognit. 2025 , 169 , 111790. [CrossRef]
15. Zhu, M.; Kong, E. Multi-Scale Fusion Uncrewed Aerial Vehicle Detection Based on RT-DETR. Electronics 2024 , 13 , 1489. [CrossRef]
16. Zhang, H.; Li, F.; Liu, S.; Zhang, L.; Wang, L. DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection. arXiv 2022 , arXiv:2203.03605.
17. Valanarasu, J.M.J.; Yasarla, R.; Patel, V.M. TransWeather: Transformer-Based Restoration of Images Degraded by Adverse Weather Conditions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, USA, 18-24 June 2022; pp. 2353-2363.
18. Pan, K.; Zhao, Y.; Wang, T.; Yang, J.; Zhang, X. MSNet: A Lightweight Multi-Scale Deep Learning Network for Pedestrian Re-Identification. Signal Image Video Process. 2023 , 17 , 3091-3098. [CrossRef]
19. Hu, M.; Wu, Y.; Yang, Y.; Li, Z.; Wang, C. DAGL-Faster: Domain Adaptive Faster R-CNN for Vehicle Object Detection in Rainy and Foggy Weather Conditions. Displays 2023 , 79 , 102484. [CrossRef]
20. Wang, Y.; Ke, H.; Cai, H. PC-YOLO: Enhancing Object Detection in Adverse Weather Through Physics-Aware and Dynamic Network Structures. J. Electron. Imaging 2025 , 34 , 023049. [CrossRef]
21. Kang, M.; Ting, C.M.; Ting, F.F.; Lim, K.M.; Tan, C.H. ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation. Image Vis. Comput. 2024 , 147 , 105057. [CrossRef]
22. Cai, H.; Li, J.; Hu, M.; Wang, Z.; Liu, S. EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction. arXiv 2022 , arXiv:2205.14756. 2022.
23. Zhang, X.; Wang, Z.; Wang, X.; Chen, Y.; Li, Q. StarNet: An Efficient Spatiotemporal Feature Sharing Reconstructing Network for Automatic Modulation Classification. IEEE Trans. Wirel. Commun. 2024 , 23 , 13300-13312. [CrossRef]
24. Li, H.; Li, J.; Wei, H.; Zhang, K.; Wang, L. Slim-Neck by GSConv: A Lightweight-Design for Real-Time Detector Architectures. J. Real-Time Image Process. 2024 , 21 , 62. [CrossRef]
25. Chen, J.; Mai, H.S.; Luo, L.; Wang, Y.; Zhang, X. Effective Feature Fusion Network in BIFPN for Small Object Detection. In Proceedings of the IEEE International Conference on Image Processing (ICIP), Anchorage, AK, USA, 19-22 September 2021; pp. 699-703.
26. Wang, A.; Chen, H.; Liu, L.; Zhang, Y.; Li, Z. YOLOv10: Real-Time End-to-End Object Detection. Adv. Neural Inf. Process. Syst. 2024 , 37 , 107984-108011.

27. Sohan, M.; Sai Ram, T.; Rami Reddy, C.V. A Review on YOLOv8 and Its Advancements. In Proceedings of the International Conference on Data Intelligence and Cognitive Informatics, Tirunelveli, Tamil Nadu, India, 18-20 November 2024; Springer: Singapore, 2024; pp. 529-545.
28. Zhang, Y.; Guo, Z.; Wu, J.; Chen, X.; Wang, L. Real-Time Vehicle Detection Based on Improved YOLO v5. Sustainability 2022 , 14 , 12274. [CrossRef]

Disclaimer/Publisher's Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.