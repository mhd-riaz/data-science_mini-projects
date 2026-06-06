# DML queries for FMCG dataset

-- involves creating database, tables and importing data from .csv file to sql
-- Step 1. create database
CREATE DATABASE fmcg_data DEFAULT CHARACTER SET = 'utf8mb4';

-- Step 2. create table(s)

-- customer(s) table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL
);

-- product(s) table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(255) NOT NULL
);

-- order(s) table
CREATE TABLE orders (
    order_id VARCHAR(255) PRIMARY KEY,
    order_placement_date DATE NOT NULL,
    customer_id INT,
    product_id INT,
    order_qty INT NOT NULL,
    agreed_delivery_date DATE NOT NULL,
    actual_delivery_date DATE NOT NULL,
    delivery_qty INT NOT NULL,
    in_full BOOLEAN,
    on_time BOOLEAN,
    otif BOOLEAN,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Step 3. enable local infile, such that we can import local data from .csv file to sql
SET GLOBAL local_infile = 1;

-- Step 4. import data from .csv file to sql

-- import customer table
LOAD DATA LOCAL INFILE '/Users/apple/Documents/PES/github/data-science_mini-projects/SQL_Tableau/data/dim_customers.csv' INTO
TABLE customers FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;
-- import product table
LOAD DATA LOCAL INFILE '/Users/apple/Documents/PES/github/data-science_mini-projects/SQL_Tableau/data/dim_products.csv' INTO
TABLE customers FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;
-- import order table
LOAD DATA LOCAL INFILE '/Users/apple/Documents/PES/github/data-science_mini-projects/SQL_Tableau/data/fact_order_lines.csv' INTO
TABLE customers FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;