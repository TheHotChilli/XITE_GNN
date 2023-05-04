"""!
@file
@brief Base class for the X-ITE pain database.
@ingroup Datasets
@addtogroup Datasets
@{
"""

class XITE():
    """!
    Base class for X-ITE dataset. Contains basic information about the dataset.
    """
    labels_invalid  = [-10,-11]
    labels_valid = [-1,-2,-3,-4,-5,-6,0,1,2,3,4,5,6]
    labels_pain = [-1,-2,-3,-4,-5,-6,1,2,3,4,5,6]
    # introduced baseline labels after slicing
    labels_base = [-100,-200,-300,-400,-500,-600,100,200,300,400,500,600]   
    classes = {
        0: "B",
        1: "pH1", 2: "pH2", 3: "pH3", 100: "BpH1", 200: "BpH2", 300: "BpH3",
        4: "tH1", 5: "tH2", 6: "tH3", 400: "BtH1", 500: "BtH2", 600: "BtH3",
        -1: "pE1", -2: "pE2", -3: "pE3", -100: "BpE1", -200: "BpE2", -300: "BpE3",
        -4: "tE1", -5: "tE2", -6: "tE3", -400: "BtE1", -500: "BtE2", -600: "BtE3"
    }
    pain_groups = ["pH", "pE", "tH", "tE"]
    base_groups = ["BpH", "BpE", "BtH", "BtE"]
    
    sample_rate = {"video": 25, "bio":1000}

"""!
@}
"""
