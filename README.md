# STAN Evaluation Script

## Setup
### Environment setup
- instructions given are a document of the steps required to run the evaluation code as they were written
- assumes that
  - the host the model will be un on is a remote linux machine with anaconda already installed
    - if this is not the case take a look at the file structure defined in the first cells of big_data_prep.ipynb and stan evaluation.ipynb to ensure that the paths specified are addressable
  - the host being used to access the remote machine is running windows

### Downloading the necessary files 
- clone the repo to the target host you plan to run the model on 
- setup the conda environment
  - definitions are stored in 'project root/conda' provided are a yaml definition useable if the device being used has internet access 
    - Use following command to install necessary libraries from the yaml file
    - conda stan create -f stan.yml
  - the folder contains all the library files required and only needs to be copied into '~/anaconda3/envs/'
  - download the csv for week 3 of may of the UGR16 data set the link to the page ugr website is here
    - https://nesg.ugr.es/nesg-ugr16/may_week3.php#INI
  - Extract the csv from the archive
    - tar -xf april_week3_csv.tar.gz

### Setup jupyter notebook
jupyter notebook needs to be run on the target host to execute the evaluation scripts

- if there is no firewall restriction, and you can freely access the notebook server over any specified port ignore the steps up too running jupyter
  - download and install putty
    - https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
  - run putty
  - On the session tab 
    - specify the IP address of the server 
    - ensure ssh connection type is selected
  - on the connection -> data tab
    - specify username
  - on the connection -> SSH -> Auth tab
    - enter the path for the '.ppk' file containing the ssh key (if you have a .pem file it needs to be converted to '.ppk')
      - If required create a ppk from an ssh key saved in a '.pem' file
        - run PuTTYgen which should be installed with putty
        - Under the 'Conversions' tab click 'Import key' and select your '.pem' file
          - You can now enter a passphrase for using the key, this requires you to reenter the passphrase whenever you try to access the key
        - With the key imported click 'Save private key'
  - on the Connection -> SSH -> Tunnels tab
    - Here we will set a port from the host to the local machine to forward
      - the source port should be the port you would like the jupyter notebook server to be on, I used 8889
      - the destination is the local address the port will be made available on your local host, I used localhost:8889
      - ensure you click the add button to save the portforward configuration
  - if you are planning on connecting to the server again be sure to save session configuration on the 'Session' tab before opening
  - with the session configured click you can now click the 'Open' button
  - enter the ppk key if previously specified
  - This connection now gives you command line access to the server and tunnels a port through SSH  to your local machine

 
- run jupyter notebook in the folder containing this repo
  - cd ~/git/stan
  - jupyter notebook --no-browser --port 8889
    - the options set the port the notebook will be run on and ensure jupyter doesn't try to locally start a browser
  - open big_data_prep.ipynb and stan_evaluation.ipynb
  - ensure that jupyter is using conda stan environment

## Running the Model
Prerequisite files should now be in place, the environment setup, and access to jupyter notebook  granted

- run the first cell in big_data_prep.ipynb to setup the initial folder
  - copy the extracted csv 'april.week3.csv.uniqblacklistremoved' to the path specified by the variable 'large_data_file'
  - run the rest of the cells in big_data_prep.ipynb


- run the first two cell of stan_evaluation.ipynb 
  - If you want to use old checkpoints, now is the time to copy them into the folder specified by the contents of 'stan_path' + "checkpoints/"
    - by default the path is "/home/ubuntu/Documents/data/stan/checkpoints"
  - Run the rest of cells seuqentially
  - after training there are two sets of cells, ones which generate and analyze a single assets worth of data and those which work with 100 assets as the STAN authors did
  - only one set of these needs to be run to save on time, but they can be run together