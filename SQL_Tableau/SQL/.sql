# Create a database - fmcg_data
CREATE DATABASE fmcg_data DEFAULT CHARACTER SET = 'utf8mb4';

# create table(s)

# customer(s) table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
);

# product(s) table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(255) NOT NULL
);

# order(s) table
CREATE TABLE orders (
    order_id VARCHAR(255) PRIMARY KEY,
    order_placement_date DATE NOT NULL,
    customer_id INT FOREIGN KEY REFERENCES customers(customer_id),
    product_id INT FOREIGN KEY REFERENCES products(product_id),
    order_qty INT NOT NULL,
    agreed_delivery_date DATE NOT NULL,
    actual_delivery_date DATE NOT NULL,
    delivery_qty INT NOT NULL,
    in_full BOOLEAN,
    on_time BOOLEAN,
    otif BOOLEAN,
);