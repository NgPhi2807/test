from pydantic import ValidationError
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .models import CustomUser, Subject, SubjectDate, StudentAttendance, SubjectStudent
from .serializers import CustomUserSerializer, SubjectSerializer, SubjectDateSerializer, StudentAttendanceSerializer, SubjectStudentSerializer,SubjectWithStudentCountSerializer,SubjectSerializerStudent,SubjectStudentInfo,SubjectDateTeacherSerializer 
from rest_framework import viewsets
from rest_framework.decorators import action
from .permissions import IsTeacher, IsStudent
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import subprocess
from django.conf import settings
import uuid
import threading
from PIL import Image
import io
import torch

class CustomUserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer

class SubjectViewSet(viewsets.ModelViewSet):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

class SubjectStudentViewSet(viewsets.ModelViewSet):
    queryset = SubjectStudent.objects.all()
    serializer_class = SubjectStudentSerializer


class LoginView(viewsets.ViewSet):
    def create(self, request, *args, **kwargs):
        username = request.data.get('username')
        password = request.data.get('password')

        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({"detail": "Invalid credentials"}, status=status.HTTP_400_BAD_REQUEST)
        
        role = user.role
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'role': role,  
        })

class UserInfoViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        user = request.user

        user_info = {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone,
            "avatar": user.avatar.url if user.avatar else None,
            "date_created": user.date_created,
        }

        return Response(user_info)

    @action(detail=False, methods=['post'], url_path='update_info')
    def update_info(self, request, *args, **kwargs):
        user = request.user
        data = request.data

        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        avatar = request.FILES.get('avatar', None)

        if not first_name or not last_name:
            return Response({"detail": "Họ và tên không được để trống."}, status=status.HTTP_400_BAD_REQUEST)
        if not email:
            return Response({"detail": "Email không được để trống."}, status=status.HTTP_400_BAD_REQUEST)

        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.phone = phone

        if avatar:
            user.avatar = avatar

        user.save()

        updated_user_info = {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone,
            "avatar": user.avatar.url if user.avatar else None,
            "date_created": user.date_created,
        }

        return Response(updated_user_info, status=status.HTTP_200_OK)
    
class TeacherSubjectsViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = SubjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.role == 'teacher':
            return Subject.objects.filter(teacher=user)  
        return Subject.objects.none() 
    
class SubjectStudentViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = SubjectStudent.objects.all()
    serializer_class = SubjectStudentSerializer

    @action(detail=True, methods=['get'], url_path='student-count')
    def student_count(self, request, pk=None):
       
        student_count = SubjectStudent.objects.filter(subject_id=pk).count()
        return Response({
            'subject_id': pk,
            'student_count': student_count
        })
class SubjectDateViewSet(viewsets.ModelViewSet):
    queryset = SubjectDate.objects.all().order_by('-current_date')
    serializer_class = SubjectDateSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        subject_date = serializer.save()
        enrolled_students = SubjectStudent.objects.filter(subject=subject_date.subject)
        attendance_list = []
        for enrollment in enrolled_students:
            attendance = StudentAttendance(
                student=enrollment.student,
                subject_date=subject_date,
                status=False,
                subject_name=subject_date.subject.name  
            )
            attendance_list.append(attendance)

        StudentAttendance.objects.bulk_create(attendance_list)

        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
class AddSubjectDateViewSet(viewsets.ModelViewSet):
    queryset = SubjectDate.objects.all()
    serializer_class = SubjectDateTeacherSerializer
    permission_classes = [IsTeacher]

    def perform_create(self, serializer):
        subject_date = serializer.save()

        subject = subject_date.subject
        student_links = SubjectStudent.objects.filter(subject=subject)

        student_attendances = [
            StudentAttendance(
                student=link.student,
                subject_date=subject_date,
                status=False  
            )
            for link in student_links
        ]
        StudentAttendance.objects.bulk_create(student_attendances)



class StudentAttendanceByDateViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = StudentAttendance.objects.all()
    serializer_class = StudentAttendanceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        queryset = super().get_queryset()
        user = self.request.user
        subject_date_id = self.request.query_params.get('subject_date_id')

        if user.role == 'teacher':
            subjects = Subject.objects.filter(teacher=user)
            queryset = queryset.filter(subject_date__subject__in=subjects)

        if subject_date_id:
            queryset = queryset.filter(subject_date__id=subject_date_id)
        return queryset
    

# views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from threading import Thread

class RecognitionThread(Thread):
    def __init__(self, subject_date_id):
        super().__init__()
        self.subject_date_id = subject_date_id

    def run(self):
        # Ví dụ: bắt đầu nhận diện, hoặc lưu trạng thái
        print(f"[INFO] Recognition started for subject_date_id={self.subject_date_id}")

class TeacherSessionViewSet(viewsets.ViewSet):

    @action(detail=False, methods=['post'])
    def start_session(self, request):
        subject_date_id = request.data.get("subject_date_id")
        if not subject_date_id:
            return Response({"error": "Missing subject_date_id"}, status=status.HTTP_400_BAD_REQUEST)

        thread = RecognitionThread(subject_date_id)
        thread.start()

        return Response({"message": f"Recognition started for subject_date_id={subject_date_id}"})

class TeacherSubjectViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    def list(self, request):
        teacher = request.user
        subjects = Subject.objects.filter(teacher=teacher)
        serializer = SubjectWithStudentCountSerializer(subjects, many=True)
        total_classes = subjects.count()

        return Response({
            "total_classes": total_classes,
            "classes": serializer.data,
        })

class StudentSubjectDateViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    def list(self, request):
        student = request.user
        if student.role != 'student':
            return Response({"detail": "Not authorized."}, status=403)

        subject_ids = SubjectStudent.objects.filter(student=student).values_list('subject_id', flat=True)

        subject_dates = SubjectDate.objects.filter(subject_id__in=subject_ids).order_by('-current_date')

        serializer = SubjectDateSerializer(subject_dates, many=True, context={'request': request})
        return Response(serializer.data)

    
class SubjectViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = SubjectStudentInfo
    permission_classes = [IsAuthenticated,IsStudent]

    def get_queryset(self):
        student = self.request.user
        return Subject.objects.filter(subjectstudent__student=student).distinct()
