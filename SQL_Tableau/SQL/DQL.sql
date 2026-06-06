-- Use the database created by DDL.sql
USE fmcg_data;

# DATA SANITIZATION

-- 1. Check orders linking to customers and products are valid.
SELECT o.order_id, o.order_placement_date, c.customer_name, c.city, p.product_name, p.category, o.order_qty, o.delivery_qty, o.in_full, o.on_time, o.otif
FROM
    orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id;

-- 2. Find orders delivered on time and in full in 2024.
--    This extracts the best-performing orders for delivery analysis.
SELECT
    order_id,
    order_placement_date,
    customer_id,
    product_id,
    order_qty,
    delivery_qty,
    in_full,
    on_time,
    otif
FROM orders
WHERE
    order_placement_date BETWEEN '2024-01-01' AND '2024-12-31'
    AND on_time = TRUE
    AND otif = TRUE;

-- 3. Find city that has high order value.
SELECT
    c.city,
    COUNT(*) AS total_orders,
    SUM(o.order_qty) AS total_order_qty,
    SUM(o.delivery_qty) AS total_delivery_qty,
    AVG(o.delivery_qty) AS avg_delivery_qty
FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
GROUP BY
    c.city
HAVING
    COUNT(*) > 10
ORDER BY total_orders DESC;

-- 4. Analyze which product category has more demand and total order quantity.
SELECT
    p.category,
    COUNT(*) AS order_count,
    SUM(o.order_qty) AS total_order_qty,
    SUM(o.delivery_qty) AS total_delivery_qty,
    AVG(o.order_qty) AS avg_order_qty
FROM orders o
    JOIN products p ON o.product_id = p.product_id
GROUP BY
    p.category
HAVING
    SUM(o.order_qty) > 100
ORDER BY total_order_qty DESC;

-- 5. Find customers who order more than the average customer.
SELECT c.customer_name, c.city, SUM(o.order_qty) AS total_order_qty
FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
GROUP BY
    c.customer_id,
    c.customer_name,
    c.city
HAVING
    SUM(o.order_qty) > (
        SELECT AVG(customer_total_qty)
        FROM (
                SELECT SUM(order_qty) AS customer_total_qty
                FROM orders
                GROUP BY
                    customer_id
            ) AS customer_totals
    );

-- 6. Flag orders with quantities above the product average.
--    This helps detect unusually large orders or potential anomalies.
SELECT o.order_id, c.customer_name, p.product_name, p.category, o.order_qty, o.delivery_qty, o.on_time, o.otif
FROM
    orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
WHERE
    o.order_qty > (
        SELECT AVG(order_qty)
        FROM orders
        WHERE
            product_id = o.product_id
    );

-- 7. Use window functions to inspect order sequence and cumulative volume.
--    This reveals customer order patterns and growing order totals.
SELECT
    o.order_id,
    o.order_placement_date,
    c.customer_name,
    p.product_name,
    p.category,
    o.order_qty,
    ROW_NUMBER() OVER (
        PARTITION BY
            o.customer_id
        ORDER BY o.order_placement_date
    ) AS order_sequence_by_customer,
    RANK() OVER (
        PARTITION BY
            o.product_id
        ORDER BY o.order_qty DESC
    ) AS rank_by_product_quantity,
    DENSE_RANK() OVER (
        PARTITION BY
            p.category
        ORDER BY o.delivery_qty DESC
    ) AS dense_rank_by_category_delivery,
    SUM(o.order_qty) OVER (
        PARTITION BY
            o.customer_id
        ORDER BY o.order_placement_date ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
    ) AS running_total_order_qty
FROM
    orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
ORDER BY c.customer_name, o.order_placement_date;