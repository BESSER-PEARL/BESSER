import re

class comparison_verifier():

    def verify_ocl(self,ocl):
        valid_pattern = r'(<|>|<=|>=|=|<>)'

        # Define a regex pattern to extract potential comparison operators
        # Matches sequences like =<, <>, ===, etc.
        potential_pattern = r'[!<>=]+'

        # Find all substrings that look like comparison operators
        potential_matches = re.findall(potential_pattern, ocl)

        # Filter out valid comparison operators
        invalid_matches = [match for match in potential_matches if not re.fullmatch(valid_pattern, match)]

        if invalid_matches:
            return False
        else:
            return True
