Welcome to the  MTech  ML1  (Sem1 ) Hackathon !


Dataset Description :
Housing price dataset of Bengaluru city is provided as train dataset (train.csv). Along with the train dataset, two more external reference datasets (avg_rent.csv & dist_from_city_centre.csv ) is given for further feature engineering.

Below are the datasets and features details


train.csv :
·      area_type: The type of the house area feature 'total_sqft' specifies.
·      availability: The availability date or availability status of the property.
·      location: The locality of the property in Bengaluru city.
·      size: The size of the housing property in BHK (or Bedrooms etc.,).
·      society: The name of the Apartment. This name is encrypted for confidentiality.
·      total_sqft: The 'area_type' area of the property.
·      bath: Number of bathrooms available in the house.
·      balcony: Number of balcony/balconies the house has.
·      price: Price of the housing property in Lakhs. (target feature)

The `test.csv` dataset contains similar information to train.csv but does not disclose the “price” feature. The price has to be predicted through your model.

Details of Additional dataset

avg_rent.csv :
·      location : The locality of the property in the bengaluru city.
·      avg_2bhk_rent : Average rent of two BHK flat in that location


dist_from_city_centre.csv :
·      location : The locality of the property in the bengaluru city.
·      dist_from_city : Distance of the location from city center.
