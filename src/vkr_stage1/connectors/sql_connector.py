from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any


class SQLConnector:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def bootstrap_demo_data(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    city TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    client_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    FOREIGN KEY(client_id) REFERENCES clients(id)
                )
                """
            )

            cur.execute("SELECT COUNT(*) FROM clients")
            if cur.fetchone()[0] == 0:
                cur.executemany(
                    "INSERT INTO clients(id, name, city) VALUES(?, ?, ?)",
                    [
                        (1, "ООО Альфа", "Москва"),
                        (2, "ИП Соколов", "Казань"),
                        (3, "ЗАО Вектор", "Санкт-Петербург"),
                    ],
                )
                cur.executemany(
                    "INSERT INTO orders(id, client_id, amount) VALUES(?, ?, ?)",
                    [
                        (1, 1, 12000.0),
                        (2, 1, 8000.0),
                        (3, 2, 5500.0),
                    ],
                )
            con.commit()

    def bootstrap_erp_data(self, size: str = "medium", reset: bool = False) -> None:
        profile = {
            "small": {"departments": 6, "employees": 40, "customers": 120, "products": 80, "orders": 250},
            "medium": {"departments": 10, "employees": 120, "customers": 800, "products": 400, "orders": 3500},
            "large": {"departments": 20, "employees": 300, "customers": 3000, "products": 1200, "orders": 15000},
        }
        if size not in profile:
            raise ValueError("size must be one of: small, medium, large")

        with self._connect() as con:
            cur = con.cursor()
            if reset:
                cur.executescript(
                    """
                    DROP TABLE IF EXISTS payments;
                    DROP TABLE IF EXISTS sales_order_items;
                    DROP TABLE IF EXISTS sales_orders;
                    DROP TABLE IF EXISTS inventory;
                    DROP TABLE IF EXISTS products;
                    DROP TABLE IF EXISTS customers;
                    DROP TABLE IF EXISTS employees;
                    DROP TABLE IF EXISTS departments;
                    DROP TABLE IF EXISTS orders;
                    DROP TABLE IF EXISTS clients;
                    """
                )

            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS departments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    city TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    department_id INTEGER NOT NULL,
                    position TEXT NOT NULL,
                    hire_date TEXT NOT NULL,
                    salary REAL NOT NULL,
                    FOREIGN KEY(department_id) REFERENCES departments(id)
                );
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    segment TEXT NOT NULL,
                    city TEXT NOT NULL,
                    registration_date TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    sku TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    warehouse TEXT NOT NULL,
                    stock_qty INTEGER NOT NULL,
                    last_update TEXT NOT NULL,
                    FOREIGN KEY(product_id) REFERENCES products(id)
                );
                CREATE TABLE IF NOT EXISTS sales_orders (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    manager_id INTEGER NOT NULL,
                    order_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES customers(id),
                    FOREIGN KEY(manager_id) REFERENCES employees(id)
                );
                CREATE TABLE IF NOT EXISTS sales_order_items (
                    id INTEGER PRIMARY KEY,
                    order_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    line_total REAL NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES sales_orders(id),
                    FOREIGN KEY(product_id) REFERENCES products(id)
                );
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY,
                    order_id INTEGER NOT NULL,
                    payment_date TEXT NOT NULL,
                    method TEXT NOT NULL,
                    amount REAL NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES sales_orders(id)
                );
                """
            )

            cur.execute("SELECT COUNT(*) FROM sales_orders")
            if cur.fetchone()[0] > 0:
                return

            cfg = profile[size]
            rng = random.Random(42)

            cities = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Нижний Новгород"]
            dep_names = ["Продажи", "Закупки", "Финансы", "HR", "Логистика", "IT", "Маркетинг", "Юридический"]
            positions = ["Менеджер", "Старший менеджер", "Аналитик", "Специалист", "Руководитель"]
            segments = ["B2B", "B2C", "Enterprise", "SMB"]
            categories = ["Электроника", "Офис", "Склад", "Сервис", "ПО"]
            statuses = ["new", "paid", "shipped", "closed", "cancelled"]
            methods = ["card", "bank_transfer", "cash"]

            departments = []
            for i in range(1, cfg["departments"] + 1):
                departments.append((i, f"{dep_names[(i - 1) % len(dep_names)]}-{i}", cities[(i - 1) % len(cities)]))
            cur.executemany("INSERT INTO departments(id, name, city) VALUES(?, ?, ?)", departments)

            base_hire = date(2017, 1, 1)
            employees = []
            for i in range(1, cfg["employees"] + 1):
                hire_dt = base_hire + timedelta(days=rng.randint(0, 3000))
                employees.append(
                    (
                        i,
                        f"Сотрудник {i}",
                        rng.randint(1, cfg["departments"]),
                        positions[rng.randint(0, len(positions) - 1)],
                        hire_dt.isoformat(),
                        round(rng.uniform(60000, 260000), 2),
                    )
                )
            cur.executemany(
                "INSERT INTO employees(id, full_name, department_id, position, hire_date, salary) VALUES(?, ?, ?, ?, ?, ?)",
                employees,
            )

            base_reg = date(2019, 1, 1)
            customers = []
            for i in range(1, cfg["customers"] + 1):
                reg_dt = base_reg + timedelta(days=rng.randint(0, 2500))
                customers.append(
                    (
                        i,
                        f"Клиент {i}",
                        segments[rng.randint(0, len(segments) - 1)],
                        cities[rng.randint(0, len(cities) - 1)],
                        reg_dt.isoformat(),
                    )
                )
            cur.executemany(
                "INSERT INTO customers(id, name, segment, city, registration_date) VALUES(?, ?, ?, ?, ?)",
                customers,
            )

            products = []
            for i in range(1, cfg["products"] + 1):
                price = round(rng.uniform(300, 200000), 2)
                products.append((i, f"SKU-{i:05d}", f"Товар {i}", categories[rng.randint(0, len(categories) - 1)], price))
            cur.executemany(
                "INSERT INTO products(id, sku, name, category, price) VALUES(?, ?, ?, ?, ?)",
                products,
            )

            inventory = []
            inv_dt = date(2026, 1, 1).isoformat()
            warehouses = ["MSK-1", "SPB-1", "KZN-1", "EKB-1"]
            inv_id = 1
            for pid in range(1, cfg["products"] + 1):
                for wh in warehouses:
                    inventory.append((inv_id, pid, wh, rng.randint(0, 200), inv_dt))
                    inv_id += 1
            cur.executemany(
                "INSERT INTO inventory(id, product_id, warehouse, stock_qty, last_update) VALUES(?, ?, ?, ?, ?)",
                inventory,
            )

            base_order = date(2023, 1, 1)
            orders = []
            order_items = []
            payments = []
            item_id = 1
            payment_id = 1
            for oid in range(1, cfg["orders"] + 1):
                order_dt = base_order + timedelta(days=rng.randint(0, 1150))
                customer_id = rng.randint(1, cfg["customers"])
                manager_id = rng.randint(1, cfg["employees"])
                status = statuses[rng.randint(0, len(statuses) - 1)]
                lines = rng.randint(1, 5)
                total = 0.0
                for _ in range(lines):
                    product_id = rng.randint(1, cfg["products"])
                    quantity = rng.randint(1, 12)
                    unit_price = products[product_id - 1][4]
                    line_total = round(quantity * unit_price, 2)
                    total += line_total
                    order_items.append((item_id, oid, product_id, quantity, unit_price, line_total))
                    item_id += 1
                total = round(total, 2)
                orders.append((oid, customer_id, manager_id, order_dt.isoformat(), status, total))

                if status in {"paid", "shipped", "closed"}:
                    payments.append(
                        (
                            payment_id,
                            oid,
                            (order_dt + timedelta(days=rng.randint(0, 20))).isoformat(),
                            methods[rng.randint(0, len(methods) - 1)],
                            total,
                        )
                    )
                    payment_id += 1

            cur.executemany(
                "INSERT INTO sales_orders(id, customer_id, manager_id, order_date, status, total_amount) VALUES(?, ?, ?, ?, ?, ?)",
                orders,
            )
            cur.executemany(
                "INSERT INTO sales_order_items(id, order_id, product_id, quantity, unit_price, line_total) VALUES(?, ?, ?, ?, ?, ?)",
                order_items,
            )
            cur.executemany(
                "INSERT INTO payments(id, order_id, payment_date, method, amount) VALUES(?, ?, ?, ?, ?)",
                payments,
            )
            con.commit()

    def get_schema_text(self) -> str:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [r[0] for r in cur.fetchall()]

            lines: list[str] = []
            for table in tables:
                cur.execute(f"PRAGMA table_info('{table}')")
                cols = cur.fetchall()
                col_text = ", ".join(f"{c['name']} {c['type']}" for c in cols)
                lines.append(f"{table}({col_text})")
        return ";\n".join(lines) + ";"

    def table_counts(self) -> dict[str, int]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [r[0] for r in cur.fetchall()]
            counts: dict[str, int] = {}
            for table in tables:
                cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
                counts[table] = int(cur.fetchone()["cnt"])
        return counts

    def execute_select(self, sql: str) -> list[dict[str, Any]]:
        normalized = sql.strip().lower()
        if not (normalized.startswith("select") or normalized.startswith("with")):
            raise ValueError("Only SELECT queries are allowed in stage 1")
        if ";" in normalized[:-1]:
            raise ValueError("Multiple SQL statements are not allowed")

        with self._connect() as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(row) for row in rows]
