from __future__ import annotations

import argparse
import json

from vkr_stage1.connectors.onec_connector import OneCConnector
from vkr_stage1.core.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real 1C HTTP endpoint")
    parser.add_argument(
        "--query",
        default="ВЫБРАТЬ ПЕРВЫЕ 1 Номенклатура ИЗ РегистрНакопления.ОстаткиТоваров",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if settings.onec_mock:
        raise ValueError("ONEC_MOCK=true. For real check set ONEC_MOCK=false in .env")

    connector = OneCConnector(
        base_url=settings.onec_base_url,
        query_path=settings.onec_query_path,
        schema_path=settings.onec_schema_path,
        username=settings.onec_username,
        password=settings.onec_password,
        mock_mode=settings.onec_mock,
    )
    response = connector.execute_query(args.query)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
