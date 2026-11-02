# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading the files a source names, for any source kind (an extraction
schema source today).

- :mod:`.locations`: local paths and S3 objects or prefixes, sized before
  they are read and read within a cap (:class:`.locations.SourceReader`,
  :func:`.locations.read_source_bytes`), through a
  :class:`.locations.SourceClientProvider`; errors as the caller's
  :class:`.locations.SourceTerms`.
- :mod:`.rdf_files`: which RDF files a path holds and the format of each
  (:func:`.rdf_files.locate`, :func:`.rdf_files.locate_s3`), and the checks
  their bytes pass before a parser sees them (:func:`.rdf_files.check_bytes`).
- :mod:`.rdf_parse`: those bytes parsed into one rdflib graph
  (:func:`.rdf_parse.parse`); the only module here that imports rdflib (the
  ``ontology`` extra), so import it only when a source runs.
"""
