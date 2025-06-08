document.getElementById('heartDiseaseForm').addEventListener('submit', function(e) {
            const button = document.getElementById('predictButton');
            const spinner = document.getElementById('loadingSpinner');
            button.disabled = true;
            spinner.classList.remove('hidden');
            setTimeout(() => {
                button.disabled = false;
                spinner.classList.add('hidden');
            }, 2000);
        });
        document.getElementById('resetButton').addEventListener('click', function() {
            const form = document.getElementById('heartDiseaseForm');
            form.reset();
            // Clear all input fields
            form.querySelectorAll('input').forEach(input => input.value = '');
            // Clear all select fields
            form.querySelectorAll('select').forEach(select => select.value = '');
        });