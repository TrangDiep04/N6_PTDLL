def generate_recommendations(input_data):
    recommendations = []
    if input_data[10] > 140:  # sysBP
        recommendations.append("Giảm muối trong khẩu phần ăn và tham khảo ý kiến bác sĩ về huyết áp.")
    if input_data[9] > 240:  # totChol
        recommendations.append("Giảm ăn chất béo bão hòa và kiểm tra cholesterol định kỳ.")
    if input_data[4] > 0:  # cigsPerDay
        recommendations.append("Cố gắng giảm hoặc bỏ hút thuốc để cải thiện sức khỏe tim mạch.")
    if input_data[12] > 30:  # BMI
        recommendations.append("Thực hiện chế độ ăn uống lành mạnh và tập thể dục thường xuyên.")
    return recommendations