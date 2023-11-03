# SapOcelExtractor

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 16.2.3.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The application will automatically reload if you change any of the source files.

## How to use the tool

1) Load the metadata tables and the process tables to be explored in the bin folder. Provide the timestamp informaiton on the tables to be analysed in the tables_data.json file, which can be found in Timestamp_JSON folder. 

2) Start the server with python app.py 

3) navigate to sap_ocel_extractor folder, and run the command 'ng serve'

4) Navigate to `http://localhost:4200/`. The tool will be sereved in the web browser. 

5) The input tables stored in the bin folder are available for extraction.

6) Type the names of the tables in the input box

7) Press Generate OCEL

8) The extracted OCEL will be saved in the server and can be used for analysis. 


## Installing necessary dependencies

1) Navigate to the root folder of the project and run the command npm install, this will install all the dependencies from the package.json file.