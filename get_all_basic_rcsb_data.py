"""
Script to fetch a basic set of data for a collection of PDB structures from RCSB.org,
and output this data to a JSON file.

Usage:
    python3 ./get_all_basic_rcsb_data.py

Output:
    full_rcsb_data.json  (~2.3 GB for all structures)

Requirements:
    pip install "rcsb-api>=1.4.0"
"""

import json
from rcsbapi.data import DataQuery
from rcsbapi.data import ALL_STRUCTURES
from rcsbapi.config import config

# Specify output file name
output_file = "full_rcsb_data.json"

# Increase some settings to speed up querying
config.DATA_API_MAX_CONCURRENT_REQUESTS = 15
config.DATA_API_REQUESTS_PER_SECOND = 30

# Prepare a Data API query to get all the basic set of data for all structures
data_query = DataQuery(
    input_type="entries",
    input_ids=ALL_STRUCTURES,
    # input_ids=["5Q5I", "4HHB", "2LGI", "1A0K"],  # Uncomment this (and comment out above "input_ids" line) for a quick test
    return_data_list=[
        "struct.title",
        "struct_keywords.pdbx_keywords",
        "struct_keywords.text",
        "exptl.method",
        "polymer_entities.rcsb_entity_source_organism.scientific_name",
        "polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name",
        "polymer_entities.rcsb_polymer_entity.rcsb_macromolecular_names_combined.name",
        "citation.title",
        "citation.pdbx_database_id_PubMed",
        "citation.pdbx_database_id_DOI",
        "citation.rcsb_is_primary",
        "pubmed.rcsb_pubmed_container_identifiers.pubmed_id",
        "pubmed.rcsb_pubmed_mesh_descriptors",
        "pubmed.rcsb_pubmed_abstract_text",
        "pubmed.rcsb_pubmed_doi",
        "pubmed.rcsb_pubmed_central_id",
        "polymer_entities.uniprots.rcsb_uniprot_protein.function.details",
        "nonpolymer_entities.nonpolymer_comp.chem_comp.name",
        "nonpolymer_entities.nonpolymer_comp.rcsb_chem_comp_synonyms.name",
    ]
)

if __name__ == "__main__":
    # Execute the query and save the results to a JSON file
    return_data = data_query.exec(progress_bar=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(return_data, f, indent=4)
    print(f"\nWrote data to output file: {output_file}")

    output_file_note = """
    OUTPUT FILE INFORMATION:
    
    The resulting JSON file will contain all the data for all IDs as a list of dictionaries/lists.
    
    For example:
    
    {
        "data": {
            "entries": [        <-- this is this list that will have all the data
                {
                    "rcsb_id": "5Q5I",    <-- this will be the unique ID for each structure, so this could be used as a key and the rest of the associated dictionary could be the value
                    "struct": {
                        "title": "PanDDA analysis group deposition -- Crystal Structure of DCLRE1A after initial refinement with no ligand modelled (structure 119)"
                    },
                    ...
                    # Note that some fields may be null
    """

    print(output_file_note)
