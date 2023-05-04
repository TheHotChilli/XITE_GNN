# X-ITE Pain Recognition using GNNs 

This project was carried out as part of my master's thesis entitled "Pain Recognition Based on Facial Expressions Using Graph Neural Networks" at the Institute of Neural Information Processing at the University of Ulm. The thesis investigates a novel approach to pain recognition that organizes [Action Units (AUs)](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) in a graph and uses [Graph Neural Networks (GNNs)](https://en.wikipedia.org/wiki/Graph_neural_network) to classify pain. The approach is evaluated performing pain classification on the [X-ITE database](https://www.nit.ovgu.de/en/Research/Databases/XITE+Pain.html) [[1]](#ref_XITE). Several scripts have emerged during the work and are made available through this repository. Most of the code is documented and a [wiki style documentation page](./docs/html/index.html) has been created using the tool doxygen. 

In the context of the work, a time-window based approach was adopted to extract facial activity based features for pain classification from facial videos. The approach consits of several substeps and builds on the works of Werner et al. [[2]](#ref_Werner), Ricken et al. [[3]](#ref_Ricken), and Tong et al.[[4]](#ref_Tong). The substeps are:
1) For the frontal face videos within the X-ITE database, the activity of 17 action units is estimated using the tool [OpenFace2](https://github.com/TadasBaltrusaitis/OpenFace/wiki). 
2) The resulting activity trajectories are then segmented into temporal windows of fixed length based on the different classes.
3) Each signal within a segment is embedded into a vector representation using statistical descriptors.
4) The embeddings of the different action units are related to each other using a graph representation.

The resulting graph representations are then processed by a Graph Neural Network architecture.

## Installation
Required python packages and their versions are listed in the `./XITE_GNN/requirements.txt` file. To install all necessary dependecies run: 
```
pip install -r requirements.txt
```
with working directory located in the corresponding directory. 

## Structure

The project consists of three main modules:
- **Datasets:** Contains various classes for handling the X-ITE dataset that are used by the other two modules. 
- **Preprocessing:** Scripts for preprocessing and feature extraction. Implements steps 2-4 of the previously described feature extraction pipeline. Contains three submodules:
    - Video_Labels
    - Slicing_and_Feature_Extraction
    - Graph_Generation
- **Training:** Scripts for definition and training of GNN models.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## References
<a name='ref_XITE'>[1] Sascha Gruss, Mattis Geiger, Philipp Werner, Oliver Wilhelm, Harald C Traue, Ayoub Al-Hamadi, and Steffen Walter. “Multi-modal signals for analyzing pain responses to thermal and electrical stimuli”. In: JoVE (Journal of Visualized Experiments) 146 (2019), e59057

<a name='ref_Werner'>[2]  Philipp Werner, Ayoub Al-Hamadi, Sascha Gruss, and Steffen Walter. “Twofold multimodal pain recognition with the X-ITE pain database”. In: 2019 8th International Conference on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW). IEEE. 2019, pp. 290–296.

<a name='ref_Ricken'>[3]  Tobias Ricken, Adrian Steinert, Peter Bellmann, Steffen Walter, and Friedhelm Schwenker. “Feature extraction: a time window analysis based on the X-ITE pain database”. In: Artificial Neural Networks in Pattern Recognition: 9th IAPR TC3 Workshop, ANNPR 2020, Winterthur, Switzerland, September 2–4, 2020, Proceedings 9. Springer. 2020, pp. 138–148.

<a name='ref_Tong'>[4] Yan Tong, Wenhui Liao, and Qiang Ji. “Facial action unit recognition by exploiting their dynamic and semantic relationships”. In: IEEE transactions on pattern analysis and machine intelligence 29.10 (2007), pp. 1683–1699.