# Where the data come from

The checked tables and saved experiments are on [Hugging Face](https://huggingface.co/datasets/rathishTharusha/leg-seir). Hugging Face hosts our files; it is not the original data source.

Case counts come from Sri Lanka's Epidemiology Unit Weekly Epidemiological Reports. Population values come from Department of Census and Statistics Census 2012 tables. The source manifest lists original URLs and SHA256 hashes. A manifest is a file list, and a hash is a file fingerprint used to detect changes.

## What is in the case table?

There are 686 report positions, called report slots, ending at Vol. 53 No. 33. Of these, 663 have complete district rows and 23 are missing. This is not a complete record of the whole 2026 calendar year.

The reports have 26 reporting divisions. We combine Ampara and Kalmunai to obtain 25 district columns. The model uses case histories and fixed population scaling. It does not use weather, measured travel or district boundaries as inputs.

A blank entry means missing data, not zero cases. Any input-and-target window that includes a missing row is excluded. We do not fill these gaps by drawing a line between known values.

The final 2026 comparison uses 18 eligible forecast starting points. Their input and target periods overlap. They are not 18 independent outbreaks.

## How counts were checked

The saved source records explain how printed counts and arithmetic repairs were accepted. Unresolved reports remain missing. Files also record cases where printed weekly and cumulative values disagree. A matching hash confirms that the file bytes match our saved copy; it does not prove that the original clinical reports are error-free or were available in real time.

The alternative printed-repeat table uses a different way of handling repeated reports. It is saved for later study. The reported LEG-SEIR results were not trained or tested on that alternative.

## File rights and access

MIT covers the team's original software. Government reports, population tables and other providers' files keep their own terms. Do not assume that the whole dataset has an MIT or unrestricted open-data license.

The public release excludes bulk original report PDFs, third-party paper PDFs, lecturer files and unused weather/geographic archives. The team keeps its original source copy separately. The download script checks original URLs against recorded hashes and reports files that have changed or disappeared.
