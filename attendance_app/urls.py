from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CustomUserViewSet, SubjectViewSet, SubjectDateViewSet, SubjectStudentViewSet, LoginView,UserInfoViewSet,TeacherSessionViewSet
from django.urls import path, include
from .views import TeacherSubjectsViewSet, StudentAttendanceByDateViewSet,TeacherSubjectViewSet,StudentSubjectDateViewSet,AddSubjectDateViewSet



router = DefaultRouter()
router.register(r'users', CustomUserViewSet)
# router.register(r'subjects', SubjectViewSet)
# router.register(r'subject-students', SubjectStudentViewSet)
# router.register(r'subject-dates', SubjectDateViewSet)
# router.register(r'attendance', StudentAttendanceViewSet)
router.register(r'login', LoginView, basename='login')  # Đăng ký route cho login
router.register(r'user_info', UserInfoViewSet, basename='user_info')  # Đăng ký viewset UserInfo
router.register(r'teacher/subjects', TeacherSubjectsViewSet, basename='teacher-subjects')
router.register(r'subjectstudent', SubjectStudentViewSet, basename='subject-student')
router.register(r'subjectdates', SubjectDateViewSet, basename='subject-date')
router.register(r'addsubjectdates', AddSubjectDateViewSet, basename='addsubject-date')

router.register(r'attendance', StudentAttendanceByDateViewSet, basename='attendance')
router.register(r'teacher-session', TeacherSessionViewSet, basename='teacher-session')
router.register(r'teacher/classes', TeacherSubjectViewSet, basename='teacher-classes')
router.register(r'students', StudentSubjectDateViewSet, basename='students')
router.register(r'studentsubjects', SubjectViewSet, basename='subject')

urlpatterns = [
    path('', include(router.urls)),
]
