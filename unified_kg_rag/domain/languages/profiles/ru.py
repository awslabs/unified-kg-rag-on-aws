# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Russian."""

from unified_kg_rag.domain.languages.codes import LanguageCode
from unified_kg_rag.domain.languages.profile import LanguageProfile

PROFILE = LanguageProfile(
    code=LanguageCode.RU,
    name="Russian",
    script="cyrillic",
    analyzer="russian",
    decline=(
        "Не удалось найти в доступных данных информацию, позволяющую ответить "
        "на этот вопрос."
    ),
    no_value_words=("неизвестно", "нет"),
    # Nominative and genitive, and the usual abbreviations.
    month_words={
        "January": ("январь", "января", "янв"),
        "February": ("февраль", "февраля", "фев"),
        "March": ("март", "марта", "мар"),
        "April": ("апрель", "апреля", "апр"),
        "May": ("май", "мая"),
        "June": ("июнь", "июня", "июн"),
        "July": ("июль", "июля", "июл"),
        "August": ("август", "августа", "авг"),
        "September": ("сентябрь", "сентября", "сен", "сент"),
        "October": ("октябрь", "октября", "окт"),
        "November": ("ноябрь", "ноября", "ноя"),
        "December": ("декабрь", "декабря", "дек"),
    },
    # The year mark: "2024 г.", "2024 года".
    date_fillers=("г", "года", "год"),
)
