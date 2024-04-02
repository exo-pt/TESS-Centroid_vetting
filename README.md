# TESS centroid_vetting

Attempt to make a centroid vetting function graphically similar to the one used in the DV reports of TCE's produced by the TESS Science Processing Pipeline.<br/>
This centroid presentation allows to identify the source of the dips in a lightcurve, and discard eventual false positives, but was limited to targets with Threshold Crossing Events (TCE).

![Img](https://github.com/exo-pt/TESS-Centroid_vetting/blob/main/tesscentroidvetting_example.png?raw=true)

## Installation
Since version 1.2.0, a package is available at PyPi
So, the simplest way of installing TESS-Centroid_Vetting is using pip:

`pip install tesscentroidvetting`

## Dependencies:
- numpy
- matplotlib
- astropy
- astroquery
- scipy
- lightkurve
- ipywidgets - only needed if interact=True in show_transit_margins()

## Usage
See the notebooks in Examples directory for usage.

## Acknowledgments
Sam Lee (https://github.com/orionlee) for implementing star's proper motion correction and transit margins display function, advising and testing. 
<br><br>
mkunimoto (https://github.com/mkunimoto) for writing **TESS-Plots**, whose code was used for the PRF centroid calculation.
<br><br>
Last but not least, Nora Eisner for [Planet Hunters Coffee Chat](https://github.com/noraeisner/PH_Coffee_Chat) series, introducing python for exoplanet detection (the difference image calculation was first borrowed from there), and for [Planet Hunters TESS](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess) citizen project, whose community [forum](https://www.zooniverse.org/projects/nora-dot-eisner/planet-hunters-tess/talk) is an invaluable source of knowledge and place to exchange views between amateurs and experts.

## Contributors
Rafael Rodrigues, Sam Lee
