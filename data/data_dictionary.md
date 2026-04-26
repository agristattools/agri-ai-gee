# Data Dictionary: USDA Northern Region Uniform Soybean Tests (SoyURT)

## Source
Krause, M.D., Dias, K.O.G., Singh, A.K., and Beavis, W.D. (2022).
Using large soybean historical data to study genotype by environment variation
and identify mega-environments with the integration of genetic and non-genetic factors.
bioRxiv, doi: 10.1101/2022.04.11.487885

## License
CC BY 4.0 (free to use with citation)

## Structure: soybean_pheno.csv

| Column | Description | Type |
|--------|-------------|------|
| year | Year of trial (1989–2019) | Integer |
| location | Trial location name | Categorical (63 levels) |
| latitude | Location latitude | Numeric |
| longitude | Location longitude | Numeric |
| altitude | Location altitude | Numeric |
| trial | Trial identifier | Categorical |
| check | Is it a check variety? | yes/no |
| maturity_group | Maturity group (II or III) | Categorical |
| G | Genotype identifier | Categorical (4,257 levels) |
| eBLUE | Empirical best linear unbiased estimate of yield | Numeric (bu/ac) |
| SE | Standard error of genotype means | Numeric |
| average_planting_date | Average planting date | Date |
| average_maturity_date | Average maturity date (days after planting) | Numeric |

## Dataset Summary
- 39,006 total observations
- 4,257 experimental genotypes
- 63 locations across the northern USA
- 591 unique location-year combinations (environments)
- 31 years of data (1989–2019)