# Supported attributes

Below is a table showing the attributes that are supported in this API for bias evaluation. The table includes a brief description of the attributes, their possible value(s), as well as the datasets that support them.

For further details, see the FHIBE [Datasheet](https://docs.google.com/document/d/1R_vghqeNU19xyRVmLxmoaR24pS475p7ILwEYkRmvPEY/edit?usp=sharing) and [CrowdWorksheet](https://docs.google.com/document/d/1OL9r_T97ddLjvz240oBTVrTCPNnQ2ULmn0D6FnVmg7o/edit?usp=sharing). The datasheet includes details on the motivation, composition, collection methods, preprocessing/cleaning/labelling, acceptable uses, distribution, and maintenance of the FHIBE dataset. The CrowdWorksheet outlines the formulation of FHIBE's tasks, the selection and payment of subjects, the selection of image annotators, platform and infrastructure choices, and the analysis and evaluation of the dataset.

<table>
  <tr>
    <th>Attribute</th>
    <th>Description</th>
    <th>Dataset</th>
    <th>Self-reported? (Y/N)</th>
    <th>Possible values</th>
    <th>Multiple values allowed? (Y/N)</th>
  </tr>
  <tr>
    <td>pronoun</td>
    <td>Gender pronouns.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. She/her/hers</li> 
        <li>1. He/him/his</li>
        <li>2. They/them/their</li>
        <li>3. Ze/zir/zirs,</li>
        <li>4. None of the above</li>
        <li>5. Prefer not to say</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>age</td>
    <td>Age at time of image capture. Subject reports exact age in years and it is binned into one of the possible values --></td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>[18-30)</li>
        <li>[30-40)</li>
        <li>[40-50)</li>
        <li>[50-60)</li>
        <li>[60,+)</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>ancestry</td>
    <td>Ancestral regions. Regional levels are required, subregion(s) are optional. </td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li> 0. Africa: 
    <ul>
        <li> 1. Northern Africa </li> 
        <li> 2. Eastern Africa </li> 
        <li> 3. Middle Africa </li> 
        <li> 4. Southern Africa </li> 
        <li> 5. Western Africa </li> 
    </ul>
</li>
<li> 6. Americas:
    <ul>
        <li> 7. Caribbean </li>
        <li> 8. Central America </li>
        <li> 9. South America </li>
        <li> 10. Northern America </li>
    </ul>
</li>
<li> 11. Asia: 
    <ul>
        <li> 12. Central Asia </li>
        <li> 13. Eastern Asia </li>
        <li> 14. South-eastern Asia </li>
        <li> 15. Southern Asia </li>
        <li> 16. Western Asia </li>
    </ul>
</li>
<li> 17. Europe: 
    <ul>
        <li> 18. Eastern Europe </li>
        <li> 19. Northern Europe </li>
        <li> 20. Southern Europe </li>
        <li> 21. Western Europe </li>
    </ul>
</li>

<li> 22. Oceania 
    <ul>
        <li> 23. Australia and New Zealand </li>
        <li> 24. Melanesia </li> 
        <li> 25. Micronesia </li>
        <li> 26. Polynesia </li>
    </ul>
</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>scene</td>
    <td>The setting of the photo.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
          <li> 0. Outdoor: Water, ice, snow </li>
          <li> 1. Outdoor: Mountains, hills, desert, sky </li>
          <li> 2. Outdoor: Forest, field, jungle </li>
          <li> 3. Outdoor: Man-made elements </li>
          <li> 4. Outdoor: Transportation </li>
          <li> 5. Outdoor: Cultural or historical building/place </li>
          <li> 6. Outdoor: Sports fields, parks, leisure spaces </li>
          <li> 7. Outdoor: Industrial and construction </li>
          <li> 8. Outdoor: Houses, cabins, gardens, and farms </li>
          <li> 9. Outdoor: Commercial buildings, shops, markets, cities, and towns </li>
          <li> 10. Indoor: Shopping and dining </li>
          <li> 11. Indoor: Workplace </li>
          <li> 12. Indoor: Home or hotel </li>
          <li> 13. Indoor: Transportation </li>
          <li> 14. Indoor: Sports and leisure </li>
          <li> 15. Indoor: Cultural </li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>camera_position</td>
    <td>Position of the camera relative to the primary subject.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Typical: Camera was at the primary subject’s eye line.</li>
        <li>1. Atypical High: Camera was above the primary subject’s eye line.</li>
        <li>2. Atypical Low: Camera was below the primary subject’s eye line.</li>
      </ul>  
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>camera_distance</td>
    <td>Derived from the face bounding box.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. CD I: Face height is in the range 10–49 pixels.</li>
        <li>1. CD II: Face height is in the range 50–299 pixels.</li>
        <li>2. CD III: Face height is 300–899 pixels.</li>
        <li>3. CD IV: Face height is 900–1499 pixels.</li>
        <li>4. CD V: Face height is 1500+ pixels.</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>natural_skin_color</td>
    <td>Natural skin color.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. [102, 78, 65]: Dark (Fitzpatrick Type VI)</li>
        <li>1. [136, 105, 81]: Brown (Fitzpatrick Type V)</li>
        <li>2. [164, 131, 103]: Tan (Fitzpatrick Type IV)</li>
        <li>3. [175, 148, 120]: Intermediate (Fitzpatrick Type III)</li>
        <li>4. [189, 163, 137]: Light (Fitzpatrick Type II)</li>
        <li>5. [198, 180, 157]: Very light (Fitzpatrick Type I)</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>apparent_skin_color</td>
    <td>Perceived skin color in the image.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. [102, 78, 65]: Dark (Fitzpatrick Type VI)</li>
        <li>1. [136, 105, 81]: Brown (Fitzpatrick Type V)</li>
        <li>2. [164, 131, 103]: Tan (Fitzpatrick Type IV)</li>
        <li>3. [175, 148, 120]: Intermediate (Fitzpatrick Type III)</li>
        <li>4. [189, 163, 137]: Light (Fitzpatrick Type II)</li>
        <li>5. [198, 180, 157]: Very light (Fitzpatrick Type I)</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>hairstyle</td>
    <td>Head hairstyle.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Buzz cut</li>
        <li>2. Short</li>
        <li>3. Up (Short)</li>
        <li>4. Half-up (Short)</li>
        <li>5. Down (Short)</li>
        <li>6. Not listed(Short)</li>
        <li>7. Medium</li>
        <li>8. Up (Medium)</li>
        <li>9. Half-up (Medium)</li>
        <li>10. Down (Medium)</li>
        <li>11. Not listed(Medium)</li>
        <li>12. Long</li>
        <li>13. Up (Long)</li>
        <li>14. Half-up (Long)</li>
        <li>15. Down (Long)</li>
        <li>16. Not listed(Long)</li>
        <li>17. Not listed</li>
        <li>18. Report string</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>natural_hair_type</td>
    <td>Natural head hair type.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Straight</li>
        <li>2. Wavy</li>
        <li>3. Curly</li>
        <li>4. Kinky-coily</li>
        <li>5. Not listed</li>
        <li>6. Report string</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>apparent_hair_type</td>
    <td>Apparent head hair type.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Straight</li>
        <li>2. Wavy</li>
        <li>3. Curly</li>
        <li>4. Kinky-coily</li>
        <li>5. Not listed</li>
        <li>6. Report string</li>
      </ul>
    </td>
    <td></td>
  </tr>
  <tr>
    <td>head_pose</td>
    <td>Head pose.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Typical: The absolute pitch is smaller than 30° and the absolute yaw is less than 30°</li>
        <li>1. Atypical: The absolute pitch is larger than 30° and/or the absolute yaw is larger
than 30°
.</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>action_body_pose</td>
    <td>Body pose.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Standing</li>
        <li>1. Sitting</li>
        <li>2. Walking</li>
        <li>3. Bending/bowing</li>
        <li>4. Lying down/sleeping</li>
        <li>5. Performing martial/fighting arts</li>
        <li>6. Dancing</li>
        <li>7. Running/jogging</li>
        <li>8. Crouching/kneeling</li>
        <li>9. Getting up</li>
        <li>10. Jumping/leaping</li>
        <li>11. Falling down</li>
        <li>12. Crawling</li>
        <li>13. Swimming</li>
        <li>14. Not listed</li>
        <li>15. Report string</li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>nationality</td>
    <td>Nationality.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>For full list, see: GOV.UK list of nationalities. https://www.gov.uk/government/publications/
nationalities/list-of-nationalities Accessed November 1, 2022
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>lighting</td>
    <td>Illumination.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Lighting from above the head/face </li>
        <li>1. Lighting from below the head/face</li>
        <li>2. Lighting from in front of the head/face</li>
        <li>3. Lighting from behind the head/face</li>
        <li>4. Lighting from the left of the head/face</li>
        <li>5. Lighting from the right of the head/face</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>weather</td>
    <td>Weather conditions.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Fog</li>
        <li>1. Haze</li>
        <li>2. Snow/hail</li>
        <li>3. Rain</li>
        <li>4. Humid</li>
        <li>5. Cloud</li>
        <li>6. Clear</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>natural_hair_color</td>
    <td>Natural head hair color(s).</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Very light blond</li>
        <li>2. Light blond</li>
        <li>3. Blond</li>
        <li>4. Dark blond</li>
        <li>5. Light brown to medium brown</li>
        <li>6. Dark brown/black</li>
        <li>7. Red</li>
        <li>8. Red blond</li>
        <li>9. Gray</li>
        <li>10. White</li>
        <li>11. Not listed</li>
        <li>12. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>apparent_hair_color</td>
    <td>Apparent head hair color(s)</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Very light blond</li>
        <li>2. Light blond</li>
        <li>3. Blond</li>
        <li>4. Dark blond</li>
        <li>5. Light brown to medium brown</li>
        <li>6. Dark brown/black</li>
        <li>7. Red</li>
        <li>8. Red blond</li>
        <li>9. Gray</li>
        <li>10. White</li>
        <li>11. Not listed</li>
        <li>12. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>facial_hairstyle</td>
    <td>Facial hairstyle.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Beard</li>
        <li>2. Mustache</li>
        <li>3. Goatee</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>natural_facial_haircolor</td>
    <td>Natural face hair color(s).</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Very light blond</li>
        <li>2. Light blond</li>
        <li>3. Blond</li>
        <li>4. Dark blond</li>
        <li>5. Light brown to medium brown</li>
        <li>6. Dark brown/black</li>
        <li>7. Red</li>
        <li>8. Red blond</li>
        <li>9. Gray</li>
        <li>10. White</li>
        <li>11. Not listed</li>
        <li>12. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>apparent_facial_haircolor</td>
    <td>Apparent face hair color(s).</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Very light blond</li>
        <li>2. Light blond</li>
        <li>3. Blond</li>
        <li>4. Dark blond</li>
        <li>5. Light brown to medium brown</li>
        <li>6. Dark brown/black</li>
        <li>7. Red</li>
        <li>8. Red blond</li>
        <li>9. Gray</li>
        <li>10. White</li>
        <li>11. Not listed</li>
        <li>12. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>natural_left_eye_color</td>
    <td>Natural left eye color.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
          <li>1. Blue</li>
          <li>2. Gray</li>
          <li>3. Green</li>
          <li>4. Hazel</li>
          <li>5. Brown</li>
          <li>6. Red and violet</li>
          <li>7. Not listed</li>
          <li>8. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>apparent_left_eye_color</td>
    <td>Apparent left eye color.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
          <li>1. Blue</li>
          <li>2. Gray</li>
          <li>3. Green</li>
          <li>4. Hazel</li>
          <li>5. Brown</li>
          <li>6. Red and violet</li>
          <li>7. Not listed</li>
          <li>8. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>natural_right_eye_color</td>
    <td>Natural right eye color.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
          <li>1. Blue</li>
          <li>2. Gray</li>
          <li>3. Green</li>
          <li>4. Hazel</li>
          <li>5. Brown</li>
          <li>6. Red and violet</li>
          <li>7. Not listed</li>
          <li>8. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>apparent_right_eye_color</td>
    <td>Apparent right eye color.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
          <li>1. Blue</li>
          <li>2. Gray</li>
          <li>3. Green</li>
          <li>4. Hazel</li>
          <li>5. Brown</li>
          <li>6. Red and violet</li>
          <li>7. Not listed</li>
          <li>8. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>facial_marks</td>
    <td>Facial marks.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Tattoos</li>
        <li>2. Birthmarks</li>
        <li>3. Scars</li>
        <li>4. Burns</li>
        <li>5. Growths</li>
        <li>6. Make-up</li>
        <li>7. Face paint</li>
        <li>8. Acne</li>
        <li>9. Not listed</li>
        <li>10. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>action_subject_object_interaction</td>
    <td>Subject-object interaction(s).</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. None</li>
        <li>1. Riding</li>
        <li>2. Driving</li>
        <li>3. Watching</li>
        <li>4. Smoking</li>
        <li>5. Eating</li>
        <li>6. Drinking</li>
        <li>7. Opening or closing</li>
        <li>8. Lifting/picking up or putting down</li>
        <li>9. Writing/drawing or painting</li>
        <li>10. Catching or throwing</li>
        <li>11. Pushing pulling or extracting</li>
        <li>12. Putting on or taking</li>
        off clothing
        <li>13. Entering or exiting</li>
        <li>14. Climbing</li>
        <li>15. Pointing at</li>
        <li>16. Shooting at</li>
        <li>17. Digging/shoveling</li>
        <li>18. Playing with pets/animals</li>
        <li>19. Playing musical instrument</li>
        <li>20. Playing</li>
        <li>21. Using an electronic device</li>
        <li>22. Cutting or chopping</li>
        <li>23. Cooking</li>
        <li>24. Fishing</li>
        <li>25. Rowing</li>
        <li>26. Sailing</li>
        <li>27. Brushing teeth</li>
        <li>28. Hitting</li>
        <li>29. Kicking</li>
        <li>30. Turning</li>
        <li>31. Not listed</li>
        <li>32. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>action_subject_subject_interaction</td>
    <td>Subject-object interaction(s).</td>
    <td>FHIBE</td>
    <td>Y</td>
    <td>
      <ul>
        <li>0. Not applicable </li>
        <li>1. Talking/listening/singing </li>
        <li>2. Watching/looking </li>
        <li>3. Grabbing </li>
        <li>4. Hitting </li>
        <li>5. Kicking </li>
        <li>6. Pushing </li>
        <li>7. Hugging/embracing </li>
        <li>8. Giving/serving or taking/receiving </li>
        <li>9. Kissing </li>
        <li>10. Lifting </li>
        <li>11. Hand shaking </li>
        <li>12. Playing with </li>
        <li>13. Not listed </li>
        <li>14. Report string</li>
      </ul>
    </td>
    <td>Y</td>
  </tr>
  <tr>
    <td>user_hour_captured</td>
    <td>Approximate time of image capture, grouped into four time windows</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>N</td>
    <td>
      <ul>
        <li>"00:00-05:59" </li>
        <li>"06:00-11:59" </li>
        <li>"12:00-17:59" </li>
        <li>"18:00-23:59" </li>
      </ul>
    </td>
    <td>N</td>
  </tr>
  <tr>
    <td>location_country</td>
    <td>The country in which the image was captured. Subjects write the country in free text. We postprocess these entries to correct for alternate spellings of the same country, mis-spellings, and map omissions to None.</td>
    <td>FHIBE, FHIBE-Face</td>
    <td>NY</td>
    <td>Free text</td>
    <td>N</td>
  </tr>
</table>
