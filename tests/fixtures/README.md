# Test Fixtures for Amulet-AI Testing

This directory contains sample images and test data used by the testing framework.

## Contents

### Sample Images (copied from dataset/test/)
- `sample_phra_somdej.jpg` - Valid Phra Somdej amulet image
- `sample_phra_rod.jpg` - Valid Phra Rod amulet image  
- `sample_phra_nang_phya.jpg` - Valid Phra Nang Phya amulet image

### OOD (Out-of-Distribution) Test Files
- `ood_random_noise.jpg` - Random noise image for OOD testing
- `ood_text_document.txt` - Text file to test non-image handling
- `ood_corrupted_image.jpg` - Corrupted image file
- `ood_non_amulet.jpg` - Image of non-amulet object (e.g., coin, jewelry)

### Test Data Files
- `malformed_request.json` - Invalid API request format
- `large_file_mock.bin` - Mock file >10MB for size limit testing
- `empty_file.jpg` - Zero-byte file for edge case testing

## Usage in Tests

These fixtures are used by:
- `tests/e2e/test_api_complete.py` - End-to-end API testing
- `tests/data/check_data_quality.py` - Data validation testing
- `tests/unit/test_model_inference.py` - Unit tests for model components
- `tests/performance/test_latency.py` - Performance benchmarking

## Maintenance

When adding new test cases:
1. Add fixture files to this directory
2. Update the test files to reference new fixtures
3. Document the purpose in this README
4. Keep fixture files small (<1MB each) to avoid bloating the repository

## File Sources

Sample images are copied from the main dataset for consistency:
```bash
# Copy commands used to create fixtures
Copy-Item "dataset/test/phra_somdej/phra_somdej_001.jpg" "tests/fixtures/sample_phra_somdej.jpg"
Copy-Item "dataset/test/phra_rod/phra_rod_001.jpg" "tests/fixtures/sample_phra_rod.jpg"
Copy-Item "dataset/test/phra_nang_phya/phra_nang_phya_001.jpg" "tests/fixtures/sample_phra_nang_phya.jpg"
```

OOD files are synthetically generated or sourced from public domain images for testing purposes only.