import krippendorff

# Each column is a rater, each row a unit (e.g., a testimonial)
data = [[1, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 0]]

alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='nominal')
print("Krippendorff's Alpha:", alpha)
