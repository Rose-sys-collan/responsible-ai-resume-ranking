import pandas as pd
import re


def normalize_to_comma_list(text):
	"""Turn a free-text field into a comma-separated list.

	Steps:
	- Normalize to string, lower-case-preserving original case is optional but we keep original case.
	- Replace common separators (newline, pipes, semicolons, slashes, bullets, ' and ', '&') with commas.
	- Split on commas, strip whitespace, drop empty entries.
	- Deduplicate while preserving order.
	- Return a single string with items joined by ', '. If no items, return empty string.
	"""
	if not isinstance(text, str):
		return ""

	s = text.strip()
	if not s:
		return ""

	# fix common mojibake for bullets (e.g. 'â€¢' produced by encoding issues)
	s = s.replace('â€¢', ',')

	# normalize separators to commas
	s = re.sub(r"[\r\n\t]+", ",", s)
	s = re.sub(r"[;|/\\•·•–—]+", ",", s)
	s = re.sub(r"\s+(?:and|&|\+)\s+", ",", s, flags=re.I)

	# split, strip, remove empties
	parts = [p.strip() for p in s.split(",")]
	parts = [p for p in parts if p]

	# deduplicate while preserving order
	seen = set()
	deduped = []
	for p in parts:
		key = p.lower()
		if key not in seen:
			seen.add(key)
			deduped.append(p)

	return ", ".join(deduped)


df = pd.read_csv("clean_dataset.csv", encoding='latin1')

# Normalize education and skills columns into comma-separated lists
changed = {}
for col in ('education', 'skills'):
	if col in df.columns:
		before_nonempty = df[col].notna().sum()
		df[col] = df[col].apply(normalize_to_comma_list)
		after_nonempty = (df[col] != '').sum()
		changed[col] = (before_nonempty, after_nonempty)

print("Normalization results (col: non-empty before -> non-empty after):")
for k, v in changed.items():
	print(f"  {k}: {v[0]} -> {v[1]}")

print(df[['education','skills']].head(20))

# Save back to CSV (overwrite the source) so downstream code uses normalized lists.
df.to_csv("cleaned_dataset.csv", index=False, encoding='latin1')