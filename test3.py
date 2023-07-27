import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rasterio as rio
import folium

import ee
from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials

ee.Authenticate()