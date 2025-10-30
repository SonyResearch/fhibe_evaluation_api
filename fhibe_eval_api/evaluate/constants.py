# SPDX-License-Identifier: Apache-2.0
"""Global variables used for evaluation.

This module contains constants used for the evaluation.
"""

DEFAULT_RANDOM_STATE = 42
MODEL_OUTPUT_FILENAME: str = "model_outputs.json"
RESULTS_IOU_FILENAME: str = "results_iou_threshold.json"
RESULTS_F1_FILENAME: str = "results_f1.json"

VALID_DATASET_NAMES = ["fhibe", "fhibe_face_crop", "fhibe_face_crop_align"]

# All attributes we can aggregate over
FHIBE_ATTRIBUTE_LIST = [
    "pronoun",
    "age",
    "ancestry",
    "scene",
    "camera_position",
    "camera_distance",
    "natural_skin_color",
    "apparent_skin_color",
    "apparent_skin_color_hue_lum",
    "hairstyle",
    "natural_hair_type",
    "apparent_hair_type",
    "head_pose",
    "action_body_pose",
    "nationality",
    "lighting",
    "weather",
    "natural_hair_color",
    "apparent_hair_color",
    "facial_hairstyle",
    "natural_facial_haircolor",
    "apparent_facial_haircolor",
    "natural_left_eye_color",
    "apparent_left_eye_color",
    "natural_right_eye_color",
    "apparent_right_eye_color",
    "facial_marks",
    "action_subject_object_interaction",
    "action_subject_subject_interaction",
    "user_hour_captured",
    "location_country",
]

# The face dataset lacks the "action_subject_subject_interaction"
# but is otherwise the same
FHIBE_FACE_ATTRIBUTE_LIST = [
    "pronoun",
    "age",
    "ancestry",
    "scene",
    "camera_position",
    "camera_distance",
    "natural_skin_color",
    "apparent_skin_color",
    "apparent_skin_color_hue_lum",
    "hairstyle",
    "natural_hair_type",
    "apparent_hair_type",
    "head_pose",
    "action_body_pose",
    "nationality",
    "lighting",
    "weather",
    "natural_hair_color",
    "apparent_hair_color",
    "facial_hairstyle",
    "natural_facial_haircolor",
    "apparent_facial_haircolor",
    "natural_left_eye_color",
    "apparent_left_eye_color",
    "natural_right_eye_color",
    "apparent_right_eye_color",
    "facial_marks",
    "action_subject_object_interaction",
    "user_hour_captured",
    "location_country",
]

# Attributes where subject or annotator could select multiple options
MULTI_SELECTION_ATTRIBUTES = [
    "pronoun",
    "ancestry",
    "nationality",
    "lighting",
    "weather",
    "natural_hair_color",
    "apparent_hair_color",
    "facial_hairstyle",
    "natural_facial_haircolor",
    "apparent_facial_haircolor",
    "natural_left_eye_color",
    "apparent_left_eye_color",
    "natural_right_eye_color",
    "apparent_right_eye_color",
    "facial_marks",
    "action_subject_object_interaction",
    "action_subject_subject_interaction",
]


ATTRIBUTES_DESCRIPTION_DICT = {
    "ancestry": (
        "Ancestry: self-reported. Subjects select from a list of 27 "
        "geographic regions. Multiple selections are allowed. "
    ),
    "natural_skin_color": (
        "Natural skin color: self-reported. Subjects select a single value "
        "from the six-point Fitzpatrick skin type scale: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;I (Very light), II (Light), "
        "III (Intermediate), IV (Tan), V (Brown), VI (Dark)"
    ),
    "natural_left_eye_color": (
        "Natural left eye color: self-reported. Multiple selections are allowed, "
        "except when 'None' is selected. "
        "Subjects select from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Blue, 2. Gray, 3. Green, 4. Hazel, "
        "5. Brown, 6. Red and violet, 7. Not listed, 8. Report string]"
    ),
    "natural_right_eye_color": (
        "Natural right eye color: self-reported. Multiple selections are allowed, "
        "except when 'None' is selected. "
        "Subjects select from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Blue, 2. Gray, 3. Green, 4. Hazel, "
        "5. Brown, 6. Red and violet, 7. Not listed, 8. Report string]"
    ),
    "natural_hair_type": (
        "Natural head hair type: self-reported. Subjects select a single value "
        "from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Straight, 2. Wavy, 3. Curly, "
        "4. Kinky-coily, 5. Not listed, 6. Report string]"
    ),
    "natural_hair_color": (
        "Natural head hair color: self-reported. Multiple selections are allowed, "
        "except when 'None' is selected. "
        "Subjects select from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Very light blond, 2. Light blond, "
        "3. Blond, 4. Dark blond, 5. Light brown to medium brown, 6. Dark brown/black, "
        "7. Red, 8. Red blond, 9. Gray, 10. White, 11. Not listed, 12. Report string]"
    ),
    "natural_facial_haircolor": (
        "Natural facial hair color: self-reported. Multiple selections are allowed, "
        "except when 'None' is selected. "
        "Subjects select from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Very light blond, 2. Light blond, "
        "3. Blond, 4. Dark blond, 5. Light brown to medium brown, 6. Dark brown/black, "
        "7. Red, 8. Red blond, 9. Gray, 10. White, 11. Not listed, 12. Report string]"
    ),
    "age": (
        "Age: self-reported age between 0 and 130. We aggregate results in "
        "five age bins: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[18, 30), [30, 40), [40, 50), [50, 60), [60, +]"
    ),
    "pronoun": (
        "Gender pronouns: self-reported. Multiple selections are allowed, "
        "except when 'None of the above' or 'Prefer not to say' is selected. "
        "Subjects select from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'She/her/hers', 'He/him/his', "
        "'They/them/their', 'Ze/zir/zirs', 'None of the above', 'Prefer not to say'"
    ),
    "nationality": (
        "Nationality: self-reported. Subjects select from a list of 225  "
        "countries, 'Not listed' or to write their own response. "
        "Multiple selections are allowed."
    ),
    "apparent_skin_color": (
        "Apparent skin color: self-reported at time of image capture. "
        "Subjects select a single value "
        "from the six-point Fitzpatrick skin type scale: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;I (Very light), II (Light), "
        "III (Intermediate), IV (Tan), V (Brown), VI (Dark)"
    ),
    "apparent_skin_color_hue_lum": (
        "Apparent skin color in terms of hue and luminance. "
        "This is calculated from the face images of each subject "
        "using the methodology developed by Thong et al. "
        "(https://arxiv.org/abs/2309.05148). Hue ranges from 0-90, "
        "and luminance ranges from 0-100. Luminance over 60 corresponds "
        "to a light skin tone (versus dark skin tone), and a hue over 55 corresponds "
        "to a yellow skin hue (versus red skin hue). "
        "Using these designations, there are four possible values "
        "for this attribute: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp; light_yellow, light_red, dark_yellow, dark_red."
    ),
    "apparent_left_eye_color": (
        "Apparent left eye color: self-reported at time of image capture. "
        "Subjects select a single value from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Blue, 2. Gray, 3. Green, 4. Hazel, "
        "5. Brown, 6. Red and violet, 7. Not listed, 8. Report string]"
    ),
    "apparent_right_eye_color": (
        "Apparent right eye color: self-reported at time of image capture. "
        "Subjects select a single value from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Blue, 2. Gray, 3. Green, 4. Hazel, "
        "5. Brown, 6. Red and violet, 7. Not listed, 8. Report string]"
    ),
    "apparent_hair_type": (
        "Apparent head hair type: self-reported at time of image capture. "
        "Subjects select a single value from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Straight, 2. Wavy, 3. Curly, "
        "4. Kinky-coily, 5. Not listed, 6. Report string]"
    ),
    "hairstyle": (
        "Head hair style: self-reported at time of image capture. "
        "Subjects select a single value from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Buzz cut, 2. Short, 3. Up (Short), "
        "4. Half-up (Short), 5. Down (Short), 6. Not listed(Short), 7. Medium, "
        "8. Up (Medium), 9. Half-up (Medium), 10. Down (Medium), "
        "11. Not listed(Medium), 12. Long, 13. Up (Long), 14. Half-up (Long), "
        "15. Down (Long), 16. Not listed(Long), 17. Not listed, 18. Report string]"
    ),
    "apparent_hair_color": (
        "Apparent head hair color: self-reported at time of image capture. "
        "Subjects select a single value from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Very light blond, 2. Light blond, "
        "3. Blond, 4. Dark blond, 5. Light brown to medium brown, 6. Dark brown/black, "
        "7. Red, 8. Red blond, 9. Gray, 10. White, 11. Not listed, 12. Report string]"
    ),
    "facial_hairstyle": (
        "Facial hairstyle. self-reported at time of image capture. "
        "Multiple selections allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Beard, 2. Mustache, 3. Goatee] "
    ),
    "apparent_facial_haircolor": (
        "Apparent facial hair color: self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Very light blond, 2. Light blond, "
        "3. Blond, 4. Dark blond, 5. Light brown to medium brown, 6. Dark brown/black, "
        "7. Red, 8. Red blond, 9. Gray, 10. White, 11. Not listed, 12. Report string]"
    ),
    "facial_marks": (
        "Facial marks: self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Tattoos, 2. Birthmarks, 3. Scars, "
        "4. Burns, 5. Growths, 6. Make-up, 7. Face paint, 8. Acne, "
        "9. Not listed, 10. Report string]"
    ),
    "action_subject_object_interaction": (
        "Subject-object interaction(s): self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. None, 1. Riding, 2. Driving, 3. Watching, "
        "4. Smoking, 5. Eating, 6. Drinking, 7. Opening or closing, "
        "8. Lifting/picking up or putting down, 9. Writing/drawing or painting, "
        "10. Catching or throwing, "
        "11. Pushing, pulling or extracting, 12. Putting on or taking off clothing, "
        "13. Entering or exiting, 14. Climbing, 15. Pointing at, 16. Shooting at, "
        "17. Digging/shoveling, 18. Playing with pets/animals, "
        "19. Playing musical instrument, 20. Playing, 21. Using an electronic device, "
        "22. Cutting or chopping, 23. Cooking, 24. Fishing, 25. Rowing, "
        "26. Sailing, 27. Brushing teeth, 28. Hitting, 29. Kicking, "
        "30. Turning, 31. Not listed, 32. Report string]"
    ),
    "action_subject_subject_interaction": (
        "Subject-subject interaction(s): self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. Not applicable, 1. Talking/listening/singing, "
        "2. Watching/looking, "
        "3. Grabbing, 4. Hitting, 5. Kicking, 6. Pushing, 7. Hugging/embracing, "
        "8. Giving/serving or taking/receiving, 9. Kissing, 10. Lifting, "
        "11. Hand shaking, 12. Playing with, 13. Not listed, 14. Report string]"
    ),
    "weather": (
        "Weather: self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. Fog, 1. Hze, 2. Snow/hail, 3. Rain, 4. Humid, "
        "5. Cloud, 6. Clear]"
    ),
    "camera_position": (
        "Camera position: self-reported at time of image capture. "
        "Subjects select a single option from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp; "
        "[0. Typical: Camera was at the primary subject’s eye line, "
        "1. Atypical High: Camera was above the primary subject’s eye line, "
        "2. Atypical Low: Camera was below the primary subject’s eye line.]"
    ),
    "scene": (
        "Image capture scene: self-reported at time of image capture. "
        "Subjects select a single option from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp; [0. Outdoor: Water, ice, snow, 1. Outdoor: "
        "Mountains, hills, desert, sky, 2. Outdoor: Forest, field, jungle, "
        "3. Outdoor: Man-made elements, 4. Outdoor: Transportation, "
        "5. Outdoor: Cultural or historical building/place, "
        "6. Outdoor: Sports fields, parks, leisure spaces, "
        "7. Outdoor: Industrial and construction, "
        "8. Outdoor: Houses, cabins, gardens, and farms, "
        "9. Outdoor: Commercial buildings, shops, markets, cities, and towns, "
        "10. Indoor: Shopping and dining, 11. Indoor: Workplace, "
        "12. Indoor: Home or hotel, 13. Indoor: Transportation, "
        "14. Indoor: Sports and leisure, 15. Indoor: Cultural]"
    ),
    "head_pose": (
        "Head pose: Obtained from human annotators. "
        "Single selection from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp; [0. Typical: The absolute pitch is smaller "
        "than 30 degrees and the absolute yaw is less than 30 degrees, "
        "1. Atypical: The absolute pitch is larger than 30 degrees and/or "
        "the absolute yaw is larger than 30 degrees]"
    ),
    "camera_distance": (
        "Camera distance. Derived from the face bounding box, one of:<br/>"
        "[0. CD I: Face height is in the range 10-49 pixels, "
        "1. CD II: Face height is in the range 50-299 pixel, "
        "2. CD III: Face height is 300-899 pixels, "
        "3. CD IV: Face height is 900-1499 pixels, "
        "4. CD V: Face height is 1500+ pixels]"
    ),
    "action_body_pose": (
        "Action body pose: self-reported at time of image capture. "
        "Subjects select a single value "
        "from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. Standing, 1. Sitting, "
        "2. Walking, 3. Bending/bowing, 4. Lying down/sleeping, "
        "5. Performing martial/fighting arts, 6. Dancing, "
        "7. Running/jogging, 8. Crouching/kneeling, 9. Getting up, "
        "10. Jumping/leaping, 11. Falling down, 12. Crawling, "
        "13. Swimming, 14. Not listed, 15. Report string]"
    ),
    "lighting": (
        "Lighting. self-reported at time of image capture. "
        "Multiple selections are allowed from the following options: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;[0. Lighting from above the head/face, "
        "1. Lighting from below the head/face, "
        "2. Lighting from in front of the head/face, "
        "3. Lighting from behind the head/face, "
        "4. Lighting from the left of the head/face, "
        "5. Lighting from the right of the head/face]"
    ),
    "user_hour_captured": (
        "User hour capture: Approximate time of image capture, grouped into "
        "four time windows: <br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;'00:00-05:59', '06:00-11:59', '12:00-17:59', "
        "'18:00-23:59'"
    ),
    "location_country": (
        "Location country: The country in which the image was captured. "
        "Reported by image subject at the time of image capture. "
        "Subjects write in the country in free text. "
    ),
}

TASK_DICT = {
    "fhibe": [
        "person_localization",
        "person_parsing",
        "keypoint_estimation",
        "face_localization",
        "body_parts_detection",
    ],
    "fhibe_face": [
        "face_encoding",
        "face_verification",
        "face_parsing",
        "face_super_resolution",
    ],
}

TASK_DESCRIPTION_DICT = {
    "person_localization": (
        "The person localization task evaluates how well a model can predict "
        "person bounding boxes in the FHIBE images. "
        "It is evaluated on the full body FHIBE dataset. "
        "Each image contains one or two people, with one  "
        "ground truth bounding box per person. "
    ),
    "keypoint_estimation": (
        "The keypoint estimation task evaluates how well a model can predict "
        "the location of major body parts in an image of a person's entire body, "
        "including their face. We adopt the 17 keypoints from the COCO dataset, "
        "and each image is only evaluated on the subset of visible keypoints. "
        "This task is evaluated on the full body FHIBE dataset. "
        "Each image contains one or two people, with one  "
        "ground truth set of keypoints per person. "
    ),
    "person_parsing": (
        "The person parsing task evalutes how well a model can predict "
        "the pixels in an image that an entire person occupies. "
        "The model is not evaluated on its ability to distinguish "
        "the various body parts, but rather the entire person mask. "
        "Each image contains one or two people, with one  "
        "ground truth mask per person. "
    ),
    "face_localization": (
        "The face detection task evaluates how well a model can predict "
        "face bounding boxes in the FHIBE images. "
        "It is evaluated on the full body FHIBE dataset. "
        "Each image contains one or two people, with one "
        "ground truth face bounding box per person. "
    ),
    "body_parts_detection": (
        "The body parts detection task evaluates how well a model can predict "
        "the existence of a subset of body parts "
        "in the FHIBE images. It is evaluated on the full body FHIBE dataset. "
        "Each image contains one or two people. The face is visible in every image, "
        "but other body parts are not necessarily. "
    ),
    "face_parsing": (
        "The face parsing task evaluates how well a model can predict "
        "the masks for various facial features. The facial body parts on which the "
        "model is evaluated are a modified version of the CelebAMask-HQ labels. "
        "It is evaluated on the FHIBE-face dataset. "
        "Each image contains a single face. "
    ),
    "face_verification": (
        "The face verification task evaluates how well a model can verify that"
        "different face images of the same person belong to the same person, "
        "as well as how well it can distinguish between "
        "face images of different people."
        "It is evaluated on the FHIBE-face dataset."
        "Each image contains a single face. "
    ),
    "face_encoding": (
        "The face encoding task evaluates how well a model can encode "
        "the image of a face. "
        "It is evaluated on the FHIBE-face dataset. "
        "Each image contains a single face. "
    ),
    "face_super_resolution": (
        "The face super resolution task evaluates how well a model can "
        "enhance the resolution and quality of a low resolution image. "
    ),
}

DATASET_DESCRIPTION_DICT = {
    "fhibe": (
        "The model was evaluated using the full-resolution FHIBE dataset. "
        "This dataset consists of ~10,000 "
        "high-resolution images comprising "
        "one or two people in a background setting. "
        "The image dimensions vary from image to image, "
        "but are typically 3-10k pixels per side. "
    ),
    "fhibe_downsampled": (
        "The model was evaluated using the downsampled FHIBE dataset. "
        "This dataset consists of ~10,000 "
        "images comprising one or two people in a background setting. "
        "Each image is resized such that the larger dimension has 2048 pixels, "
        "while maintaining the aspect ratio of the original, higher resolution image."
    ),
    "fhibe_face_crop_align": (
        "The model was evaluated using the FHIBE-face dataset. "
        "This dataset consists of ~10,000 "
        "images of a single person's face. "
        "Each image has size 512x512. "
    ),
}

METADATA_UNIVERSAL_DESCRIPTION = (
    "Accompanying the dataset is a rich set of metadata. "
    "The metadata include self-reported demographic information, "
    "such as age, ancestry, and gender pronouns of image subjects, "
    "as well as annotator-provided metadata for more objective physical attributes "
    "such as pose and subject actions. "
    "The metadata also characterize the scene via lighting and weather descriptions. "
    "Drawn annotations are also provided "
    "for person bounding boxes, face bounding boxes, "
    "full body keypoints, person segmentation masks, and face segmentation masks. "
)

TASK_METADATA_DESCRIPTION_DICT = {
    "person_localization": (
        "The metadata specific to this task include "
        "the person bounding box annotations."
    ),
    "keypoint_estimation": (
        "The metadata specific to this task include "
        "the full body keypoint annotations."
    ),
    "body_parts_detection": (
        "The metadata specific to this task include "
        "the segmentation mask annotations "
        "and the keypoint annotations, which are used to define the "
        "ground truth body parts for each person in each image. "
    ),
    "person_parsing": (
        "The metadata specific to this task include "
        "the person segmentation mask annotations."
    ),
    "face_localization": (
        "The metadata specific to this task include "
        "the face bounding box annotations. "
    ),
    "face_parsing": (
        "The metadata specific to this task include "
        "the face segmentation mask annotations. "
    ),
    "face_verification": None,
    "face_encoding": None,
    "face_super_resolution": None,
}

FITZPATRICK_TYPE_DICT = {
    (102, 78, 65): "VI",
    (136, 105, 81): "V",
    (164, 131, 103): "IV",
    (175, 148, 120): "III",
    (189, 163, 137): "II",
    (198, 180, 157): "I",
}

FITZPATRICK_RGB_DICT = {
    "VI": (102, 78, 65),
    "V": (136, 105, 81),
    "IV": (164, 131, 103),
    "III": (175, 148, 120),
    "II": (189, 163, 137),
    "I": (198, 180, 157),
}

ATTRIBUTE_CONSOLIDATION_DICT = {
    "ancestry": {
        "0. Africa": [
            "1. Northern Africa",
            "2. Eastern Africa",
            "3. Middle Africa",
            "4. Southern Africa",
            "5. Western Africa",
        ],
        "6. Americas": [
            "7. Caribbean",
            "8. Central America",
            "9. South America",
            "10. Northern America",
        ],
        "11. Asia": [
            "12. Central Asia",
            "13. Eastern Asia",
            "14. South-eastern Asia",
            "15. Southern Asia",
            "16. Western Asia",
        ],
        "17. Europe": [
            "18. Eastern Europe",
            "19. Northern Europe",
            "20. Southern Europe",
            "21. Western Europe",
        ],
        "22. Oceania": [
            "23. Australia and New Zealand",
            "24. Melanesia",
            "25. Micronesia",
            "26. Polynesia",
        ],
    },
    "nationality": {
        "227. African": [
            "2. Algerian",
            "5. Angolan",
            "20. Beninese",
            "25. Botswanan",
            "31. Burkinan",
            "33. Burundian",
            "35. Cameroonian",
            "37. Cape Verdean",
            "39. Central African",
            "40. Chadian",
            "44. Comoran",
            "45. Congolese (Congo)",
            "46. Congolese (DRC)",
            "56. Djiboutian",
            "62. Egyptian",
            "65. Equatorial Guinean",
            "66. Eritrean",
            "68. Ethiopian",
            "74. Gabonese",
            "75. Gambian",
            "78. Ghanaian",
            "86. Guinean",
            "85. Citizen of Guinea-Bissau",
            "100. Ivorian",
            "105. Kenyan",
            "139. Mosotho",
            "114. Liberian",
            "115. Libyan",
            "121. Malagasy",
            "122. Malawian",
            "125. Malian",
            "129. Mauritanian",
            "130. Mauritian",
            "138. Moroccan",
            "140. Mozambican",
            "141. Namibian",
            "147. Nigerien",
            "146. Nigerian",
            "168. Rwandan",
            "172. Sao Tomean",
            "175. Senegalese",
            "177. Citizen of Seychelles",
            "178. Sierra Leonean",
            "183. Somali",
            "184. South African",
            "186. South Sudanese",
            "192. Sudanese",
            "200. Tanzanian",
            "202. Togolese",
            "206. Tunisian",
            "211. Ugandan",
            "223. Zambian",
            "224. Zimbabwean",
        ],
        "228. North or South American": [
            "3. American",
            "7. Citizen of Antigua and Barbuda",
            "8. Argentine",
            "13. Bahamian",
            "16. Barbadian",
            "19. Belizean",
            "21. Bermudian",
            "23. Bolivian",
            "26. Brazilian",
            "36. Canadian",
            "38. Cayman Islander",
            "41. Chilean",
            "43. Colombian",
            "48. Costa Rican",
            "50. Cuban",
            "57. Dominican",
            "58. Citizen of the Dominican Republic",
            "61. Ecuadorean",
            "169. Salvadorean",
            "74. Falkland Islander",
            "128. Martiniquais",
            "131. Mexican",
            "137. Montserratian",
            "145. Nicaraguan",
            "156. Panamanian",
            "158. Paraguayan",
            "159. Peruvian",
            "164. Puerto Rican",
            "189. St Helenian",
            "190. St Lucian",
            "219. Vincentian",
            "207. Trinidadian",
            "209. Turks and Caicos Islander",
            "213. Uruguayan",
            "217. Venezuelan",
            "28. British Virgin Islander",
            "250. Virgin Islander, U.S.",
        ],
        "229. Asian": [
            "0. Afghan",
            "9. Armenian",
            "12. Azerbaijani",
            "14. Bahraini",
            "15. Bangladeshi",
            "22. Bhutanese",
            "29. Bruneian",
            "34. Cambodian",
            "42. Chinese",
            "53. Cypriot",
            "60. East Timorese",
            "76. Georgian",
            "90. Hong Konger",
            "93. Indian",
            "94. Indonesian",
            "95. Iranian",
            "96. Iraqi",
            "98. Israeli",
            "102. Japanese",
            "103. Jordanian",
            "104. Kazakh",
            "109. Kuwaiti",
            "110. Kyrgyz",
            "111. Lao",
            "113. Lebanese",
            "119. Macanese",
            "123. Malaysian",
            "124. Maldivian",
            "135. Mongolian",
            "32. Burmese",
            "143. Nepalese",
            "149. North Korean",
            "152. Omani",
            "153. Pakistani",
            "155. Palestinian",
            "71. Filipino",
            "165. Qatari",
            "173. Saudi Arabian",
            "179. Singaporean",
            "185. South Korean",
            "188. Sri Lankan",
            "197. Syrian",
            "198. Taiwanese",
            "199. Tajik",
            "201. Thai",
            "207. Turkish",
            "208. Turkmen",
            "239. Uzbek",
            "218. Vietnamese",
            "222. Yemeni",
        ],
        "230. European": [
            "1. Albanian",
            "4. Andorran",
            "11. Austrian",
            "17. Belarusian",
            "18. Belgian",
            "24. Citizen of Bosnia and Herzegovina",
            "30. Bulgarian",
            "49. Croatian",
            "53. Cypriot",
            "54. Czech",
            "55. Danish",
            "67. Estonian",
            "69. Faroese",
            "72. Finnish",
            "73. French",
            "77. German",
            "79. Gibraltarian",
            "80. Greek",
            "81. Greenlandic",
            "91. Hungarian",
            "92. Icelandic",
            "97. Irish",
            "99. Italian",
            "116. Liechtenstein citizen",
            "112. Latvian",
            "117. Lithuanian",
            "118. Luxembourger",
            "120. Macedonian",
            "126. Maltese",
            "133. Moldovan",
            "134. Monegasque",
            "136. Montenegrin",
            "59. Dutch",
            "151. Norwegian",
            "161. Polish",
            "162. Portuguese",
            "166. Romanian",
            "167. Russian",
            "170. Sammarinese",
            "176. Serbian",
            "180. Slovak",
            "181. Slovenian",
            "187. Spanish",
            "220. Wallisian",
            "195. Swedish",
            "196. Swiss",
            "212. Ukrainian",
            "27. British",
            "215. Vatican citizen",
        ],
        "231. Oceanian": [
            "10. Australian",
            "47. Cook Islander",
            "70. Fijian",
            "83. Guamanian",
            "107. Citizen of Kiribati",
            "127. Marshallese",
            "132. Micronesian",
            "144. New Zealander",
            "148. Niuean",
            "160. Pitcairn Islander",
            "154. Palauan",
            "157. Papua New Guinean",
            "171. Samoan",
            "182. Solomon Islander",
            "203. Tongan",
            "210. Tuvaluan",
            "216. Citizen of Vanuatu",
            "220. Wallisian",
        ],
        "232. Other": ["191. Stateless", "225. Not listed", "226. Report string"],
    },
}
