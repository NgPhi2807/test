from rest_framework import serializers
from .models import CustomUser, Subject, SubjectDate, StudentAttendance, SubjectStudent


class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = '__all__'

class SubjectSerializer(serializers.ModelSerializer):
    teacher_name = serializers.SerializerMethodField()   
    class Meta:
        model = Subject
        fields = '__all__'
    def get_teacher_name(self, obj):
        return f"{obj.teacher.first_name} {obj.teacher.last_name}" if obj.teacher else None    

# class SubjectStudentSerializer(serializers.ModelSerializer):
#     subject = SubjectSerializer()  
#     class Meta:
#         model = SubjectStudent
#         fields = '__all__'

class SubjectStudentSerializer(serializers.ModelSerializer):
    subject = serializers.PrimaryKeyRelatedField(queryset=Subject.objects.all()) 
    subject_name = serializers.CharField(source='subject.name', read_only=True)  
    class Meta:
        model = SubjectStudent
        fields = ['id', 'subject', 'subject_name', 'student']  
class SubjectDateSerializer(serializers.ModelSerializer):
    name_subject = serializers.SerializerMethodField()  
    class Meta:
        model = SubjectDate
        fields = ['id', 'subject', 'current_date','name_subject','status']
    def get_name_subject(self, obj):
        return obj.subject.name if obj.subject else None  

class StudentAttendanceSerializer(serializers.ModelSerializer):
    student = serializers.StringRelatedField()  
    subject_date = SubjectDateSerializer()  
    name_subject = serializers.SerializerMethodField()  
    
    class Meta:
        model = StudentAttendance
        fields = ['student', 'subject_date','name_subject','status']

    def get_name_subject(self, obj):
        return obj.subject_date.subject.name if obj.subject_date else None

class SubjectWithStudentCountSerializer(serializers.ModelSerializer):
    student_count = serializers.SerializerMethodField()

    class Meta:
        model = Subject
        fields = ['id', 'name', 'student_count']

    def get_student_count(self, obj):
        return SubjectStudent.objects.filter(subject=obj).count()
    
class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email']

class SubjectDateSerializer(serializers.ModelSerializer):
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    start_time = serializers.TimeField(source='subject.time_start', read_only=True)
    end_time = serializers.TimeField(source='subject.time_end', read_only=True)
    attendance_status = serializers.SerializerMethodField()
    
    class Meta:
        model = SubjectDate
        fields = ['id', 'current_date', 'status', 'subject_name', 'start_time', 'end_time', 'attendance_status']

    def get_attendance_status(self, obj):
        user = self.context['request'].user
        attendance = StudentAttendance.objects.filter(subject_date=obj, student=user).first()
        if attendance:
            return attendance.status
        return None 

class SubjectDateTeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectDate
        fields = ['id', 'subject', 'current_date', 'status']



class StudentAttendanceSerializer(serializers.ModelSerializer):
    student = serializers.CharField(source='student.username')  # username hoặc mã sinh viên
    first_name = serializers.CharField(source='student.first_name')
    last_name = serializers.CharField(source='student.last_name')
    status = serializers.BooleanField()
    subject_date = serializers.PrimaryKeyRelatedField(read_only=True)
    current_date = serializers.DateField(source='subject_date.current_date', read_only=True)
    class Meta:
        model = StudentAttendance
        fields = ['student', 'first_name', 'last_name', 'status', 'subject_date', 'current_date']

class SubjectSerializerStudent(serializers.ModelSerializer):
    subject_dates = serializers.SerializerMethodField()
    attendances = serializers.SerializerMethodField()

    class Meta:
        model = Subject
        fields = ['id', 'name', 'time_start', 'time_end', 'date_start', 'date_end', 'subject_dates', 'attendances']

    def get_subject_dates(self, obj):
        dates = SubjectDate.objects.filter(subject=obj)
        return SubjectDateSerializer(dates, many=True).data

    def get_attendances(self, obj):
        student = self.context['request'].user
        attendances = StudentAttendance.objects.filter(student=student, subject_date__subject=obj)
        return StudentAttendanceSerializer(attendances, many=True).data
    

class SubjectStudentInfo(serializers.ModelSerializer):
    teacher = TeacherSerializer(read_only=True)

    class Meta:
        model = Subject
        fields = ['id', 'name', 'teacher', 'time_start', 'time_end', 'date_start', 'date_end']