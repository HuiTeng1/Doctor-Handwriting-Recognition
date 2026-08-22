"""
Structured source-of-truth spec for every step of every pipeline. This is where
"mandatory classification" is enforced: every operation in pipelines.py must have a
matching entry here (checked with verify_coverage() when the module loads) - a missing
classification/annotation raises an error immediately, it never quietly becomes an
unowned assumption.

Classification definitions:
  A - Original paper: explicitly given in the paper's body/table/figure
  B - Verified library default: paper doesn't give a parameter, so a verified/tested
      library default value or implementation is used
  C - Additional external reference: paper doesn't give a parameter, so an external
      method with a clear academic source is used
  D - Interpretation from paper: an inference from the paper's wording, not a parameter
      the authors explicitly specified
  (E - Unsupported/unresolved is not allowed to appear; if a step would land on E, it
  must be reported to the user first, not coded directly)
"""
from dataclasses import dataclass

OTSU_REF = "Otsu, N. (1979), \"A Threshold Selection Method from Gray-Level Histograms,\" IEEE Trans. SMC, vol. SMC-9, no.1, pp.62-66"
CANNY_MEDIAN_REF = ("Ghahremannezhad, H., Shi, H., & Liu, C. (2020), \"A New Adaptive Bidirectional "
                     "Region-of-Interest Detection Method for Intelligent Traffic Video Analysis,\" "
                     "2020 IEEE AIKE, pp.17-24, DOI:10.1109/AIKE48582.2020.00012 "
                     "(verified by reading full PDF; paper's domain is traffic video, formula borrowed "
                     "purely as a general Canny-threshold technique, not a handwriting-recognition source)")
PILLOW_UNSHARP_REF = "Pillow 12.2.0, PIL.ImageFilter.UnsharpMask.__init__ verified default signature (radius=2, percent=150, threshold=3)"
PILLOW_GRAYSCALE_REF = "Pillow 12.2.0, PIL.Image.convert('L') verified internal formula (ITU-R 601-2: L = R*299/1000 + G*587/1000 + B*114/1000)"
TORCHVISION_RESIZE_REF = "torchvision 0.20.1+cu121, transforms.Resize.__init__ verified default interpolation=InterpolationMode.BILINEAR"
SKIMAGE_ZHANG_REF = ("scikit-image 0.26.0, skimage.morphology.skeletonize verified docstring: "
                      "\"Zhang's algorithm [Zha84] ... is the default for 2D\", citing "
                      "T.Y. Zhang and C.Y. Suen, Communications of the ACM, March 1984, Vol.27, No.3")

CLHAA_PAPER = "CLHAA paper, Section III-B-2 \"Preprocessing\""
FAJARDO_PAPER = "Fajardo et al. 2019, Section B \"Preprocessing and Cursive Handwriting Samples\""
BENITEZ_PAPER = "Benitez et al. 2024, \"All Preprocessing (White)\" combination"


@dataclass(frozen=True)
class Step:
    name: str
    parameter: str
    source_class: str  # 'A' | 'B' | 'C' | 'D'
    source: str
    note: str = ""

    def __post_init__(self):
        if self.source_class not in ("A", "B", "C", "D"):
            raise ValueError(
                f"Step '{self.name}': source_class={self.source_class!r} is outside the allowed range (A/B/C/D). "
                f"If this step can only land on E (no supporting source), don't register it here - report it to the user first."
            )


PIPELINE_SPECS = {
    "baseline": [
        Step("no_preprocessing", "original image, unchanged", "A",
             "existing test code (no research preprocessing applied, only the input conversion built into CRNN/TrOCR is kept)"),
    ],
    "clhaa": [
        Step("pixel_scaling", "[0,255] -> [0,1]", "A", CLHAA_PAPER,
             "functionally redundant with the ToTensor()/TrOCRProcessor normalization already built into the existing test code, not implemented separately as an on-disk image operation"),
        Step("resize_aspect_ratio", "preserve aspect ratio, long side=224", "A", CLHAA_PAPER),
        Step("resize_interpolation", "bilinear", "B", TORCHVISION_RESIZE_REF),
        Step("white_padding", "pad to 224x224, fill=255", "A", CLHAA_PAPER),
        Step("unsharp_technique", "unsharp mask using Gaussian blur", "A", CLHAA_PAPER),
        Step("unsharp_radius", "2", "B", PILLOW_UNSHARP_REF),
        Step("unsharp_percent", "150", "B", PILLOW_UNSHARP_REF),
        Step("unsharp_threshold", "3", "B", PILLOW_UNSHARP_REF),
        Step("grayscale_formula", "0.2989R+0.5870G+0.1140B", "A", CLHAA_PAPER,
             "matches Pillow's verified internal .convert('L') formula, the two corroborate each other"),
        Step("step_order", "grayscale applied after sharpening", "A", CLHAA_PAPER,
             "the paper's body explicitly states \"applied after the sharpening step\""),
        Step("final_size", "224x224", "A", CLHAA_PAPER, "produced directly by resize+padding, no second resize"),
    ],
    "fajardo": [
        Step("binarization_method", "Otsu", "C", OTSU_REF, "the Fajardo paper only says \"binary image\", doesn't specify the method"),
        Step("thinning_algorithm", "Zhang-Suen", "A", FAJARDO_PAPER),
        Step("thinning_implementation", "skimage.morphology.skeletonize(method='zhang')", "B", SKIMAGE_ZHANG_REF),
        Step("height_normalization", "64 px", "A", FAJARDO_PAPER),
        Step("width_handling", "variable, preserve aspect ratio", "D", FAJARDO_PAPER,
             "inferred from the paper's phrase \"sequence capacity adjusts depending on length\", not a parameter the authors stated explicitly"),
        Step("resize_interpolation", "bilinear", "B", TORCHVISION_RESIZE_REF),
    ],
    "benitez": [
        Step("grayscale", "RGB -> L", "A", BENITEZ_PAPER, "the source image is already grayscale, so this step is a no-op in our pipeline"),
        Step("binarization_method", "Otsu", "C", OTSU_REF, "the Benitez paper only mentions a binarization stage, doesn't specify the method"),
        Step("canny_technique", "Canny edge detection", "A", BENITEZ_PAPER,
             "the paper reports this combination at ~90.76% accuracy, CER 1.79%, WER 3.61%; no specific thresholds given"),
        Step("canny_lower", "0.66 x median", "C", CANNY_MEDIAN_REF, "verified in person by reading the full PDF text"),
        Step("canny_upper", "1.33 x median", "C", CANNY_MEDIAN_REF, "verified in person by reading the full PDF text"),
        Step("inversion", "255 - pixel", "A", BENITEZ_PAPER, "the operation itself has no parameter choice, it's a mathematical definition"),
    ],
}


def verify_coverage(pipeline_name, step_names_used):
    """Called after each pipeline function in pipelines.py finishes running, confirming
    every step it actually executed is registered in the spec, and that every step
    registered in the spec was actually used too (bidirectional check, so the spec and
    the implementation can never silently diverge)."""
    declared = {s.name for s in PIPELINE_SPECS[pipeline_name]}
    used = set(step_names_used)
    missing_from_spec = used - declared
    unused_in_spec = declared - used
    if missing_from_spec:
        raise ValueError(
            f"[{pipeline_name}] The following steps ran in code but have no source classification registered in spec.py: "
            f"{missing_from_spec}. Implementing a step without a classification isn't allowed - add an A/B/C/D entry to spec.py first."
        )
    if unused_in_spec:
        raise ValueError(
            f"[{pipeline_name}] spec.py registers the following steps, but the code never actually ran them: "
            f"{unused_in_spec}. The spec and the implementation must match exactly."
        )


def spec_table_rows():
    """For results/spec_table.csv: flattens PIPELINE_SPECS into table rows."""
    rows = []
    for pipeline_name, steps in PIPELINE_SPECS.items():
        for s in steps:
            rows.append({
                "pipeline": pipeline_name,
                "step": s.name,
                "parameter": s.parameter,
                "source_class": s.source_class,
                "source": s.source,
                "note": s.note,
            })
    return rows
