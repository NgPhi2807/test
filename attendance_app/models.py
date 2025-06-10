from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('teacher', 'Teacher'),
        ('student', 'Student'),
    ]
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='student')
    phone = models.CharField(max_length=20, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True, default='avatars/avatar.jpg')
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.username} - {self.role}"

    @property
    def full_name(self):
        return self.username  

    def get_short_name(self):
        return self.username

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

class Subject(models.Model):
    name = models.CharField(max_length=100)
    teacher = models.ForeignKey(CustomUser, on_delete=models.CASCADE, limit_choices_to={'role': 'teacher'})
    time_start = models.TimeField()
    time_end = models.TimeField()
    date_start = models.DateField()
    date_end = models.DateField()

    def __str__(self):
        return self.name

class SubjectStudent(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    student = models.ForeignKey(CustomUser, on_delete=models.CASCADE, limit_choices_to={'role': 'student'})

    class Meta:
        unique_together = ('subject', 'student')
    def __str__(self):
        return f"{self.student.username} - {self.subject.name}"

class SubjectDate(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    current_date = models.DateField()
    status = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.subject.name} - {self.current_date}"



class StudentAttendance(models.Model):
    student = models.ForeignKey(CustomUser, on_delete=models.CASCADE, limit_choices_to={'role': 'student'})
    subject_date = models.ForeignKey(SubjectDate, on_delete=models.CASCADE)
    status = models.BooleanField(default=False)  

    class Meta:
        unique_together = ('student', 'subject_date')
    def __str__(self):
        return f"{self.student.username} - {self.subject_date}"
