## Overview

1)  This repository contains the code for the SAP OCEL extractor tool.
    To use the tool, please download and load the following metadata tables in the path,
    bin/SAP_metadata_tables -->
    DD02V , DD03L , DD03T , DD07T , T003T , TCDOB , TCDOBT , TSTCT

    You can load the process related tables in the bin/ProcessTables repository.

    Further instructions to start the tool are in the README of sap_ocel_extractor.

## Tips to extract the metadata tables. 

1) DDO3L: 
    SELECT * FROM USER.DD03L

2) TCDOB:
    SELECT OBJECT , TABNAME FROM USER.TCDOB

3) TCDOBT: 
    SELECT * FROM USER.TCDOBT WHERE SPRAS = 'E'

4) DD07T:
    SELECT DOMNAME , DDLANGUAGE , DDTEXT, DOMVALUE_L FROM USER.DD07T WHERE DDLANGUAGE = 'E'

5) T003T:
    SELECT * FROM USER.T003T WHERE SPRAS = 'E'

6) TSTCT:
    SELECT * FROM USER.TSTCT WHERE SPRSL = 'E'

7) DD03T:
    SELECT * FROM USER.DD03T WHERE DDLANGUAGE = 'E'

8) DD02V:
    SELECT TABNAME , DDLANGUAGE , TABCLASS , DDTEXT  FROM USER.DD02V WHERE DDLANGUAGE = 'E'
